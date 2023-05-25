#ifdef GL_ES
precision mediump float;
#endif

// Phong related variables
uniform sampler2D uSampler;
uniform vec3 uKd;
uniform vec3 uKs;
uniform vec3 uLightPos;
uniform vec3 uCameraPos;
uniform vec3 uLightIntensity;

varying highp vec2 vTextureCoord;
varying highp vec3 vFragPos;
varying highp vec3 vNormal;

// Shadow map related variables
#define NUM_SAMPLES 20
#define BLOCKER_SEARCH_NUM_SAMPLES NUM_SAMPLES
#define PCF_NUM_SAMPLES NUM_SAMPLES
#define NUM_RINGS 10

#define EPS 1e-3
#define PI 3.141592653589793
#define PI2 6.283185307179586

uniform sampler2D uShadowMap;

varying vec4 vPositionFromLight;

highp float rand_1to1(highp float x ) { 
  // -1 -1
  return fract(sin(x)*10000.0);
}

highp float rand_2to1(vec2 uv ) { 
  // 0 - 1
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract(sin(sn) * c);
}

float unpack(vec4 rgbaDepth) {
    const vec4 bitShift = vec4(1.0, 1.0/256.0, 1.0/(256.0*256.0), 1.0/(256.0*256.0*256.0));
    return dot(rgbaDepth, bitShift);
}

vec2 poissonDisk[NUM_SAMPLES];
//泊松圆盘采样
void poissonDiskSamples( const in vec2 randomSeed ) {

  float ANGLE_STEP = PI2 * float( NUM_RINGS ) / float( NUM_SAMPLES );
  float INV_NUM_SAMPLES = 1.0 / float( NUM_SAMPLES );

  float angle = rand_2to1( randomSeed ) * PI2;
  float radius = INV_NUM_SAMPLES;
  float radiusStep = radius;

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( cos( angle ), sin( angle ) ) * pow( radius, 0.75 );
    radius += radiusStep;
    angle += ANGLE_STEP;
  }
}
//均匀圆盘采样
void uniformDiskSamples( const in vec2 randomSeed ) {

  float randNum = rand_2to1(randomSeed);
  float sampleX = rand_1to1( randNum ) ;
  float sampleY = rand_1to1( sampleX ) ;

  float angle = sampleX * PI2;
  float radius = sqrt(sampleY);

  for( int i = 0; i < NUM_SAMPLES; i ++ ) {
    poissonDisk[i] = vec2( radius * cos(angle) , radius * sin(angle)  );

    sampleX = rand_1to1( sampleY ) ;
    sampleY = rand_1to1( sampleX ) ;

    angle = sampleX * PI2;
    radius = sqrt(sampleY);
  }
}

float findBlocker( sampler2D shadowMap,  vec2 uv, float zReceiver ) {
  int blockerNum = 0;           //着色点对应到shadow map中后，周围一圈邻域内为blocker的点的数量
  float blocker_depth = 0.0;    //blocker的深度
  float shadowMapSize = 2048.0; //shadow map的分辨率
  float stride = 20.0;          //采样的步长
  float filterRange = stride / shadowMapSize; //滤波窗口的范围

  //泊松圆盘采样得到采样点
  poissonDiskSamples(uv);

  //均匀圆盘采样得到采样点
  //uniformDiskSamples(uv);

  //判断着色点对应到shadow map中后，邻域中的点是否为blocker，如果是就累加
  for(int i = 0; i < NUM_SAMPLES; i++){
    float shadow_depth = unpack(texture2D(shadowMap,uv+ poissonDisk[i] * filterRange)); 
    if(zReceiver > shadow_depth + 0.01){
      blockerNum++;
      blocker_depth += shadow_depth;
    }
  }

  if(blockerNum == 0){
    return 1.0;
  }

  blocker_depth = blocker_depth / float(blockerNum);
	return blocker_depth;
}

float PCF(sampler2D shadowMap, vec4 coords) {
  float stride = 10.0;           //定义步长
  float shadowMapSize = 2048.0;  //shadowmap分辨率
  float visibility1 = 0.0;        //初始可见项
  float cur_depth = coords.z;    //卷积范围内当前点的深度
  float filterRange = stride / shadowMapSize; //滤波窗口的范围

  //泊松圆盘采样得到采样点
  poissonDiskSamples(coords.xy);

  //均匀圆盘采样得到采样点
  //uniformDiskSamples(coords.xy);

  //对每个点进行比较深度值并累加
  for(int i = 0; i < NUM_SAMPLES; i++){
    float shadow_depth = unpack(texture2D(shadowMap,coords.xy + poissonDisk[i] * filterRange));
    float res = (cur_depth < shadow_depth + EPS) ? 1.0 : 0.0;
    visibility1 += res;
  }

  if(NUM_SAMPLES == 0){
    return 1.0;
  }

  //返回均值
  visibility1 = visibility1 / float(NUM_SAMPLES);
  return visibility1;
}

float PCSS(sampler2D shadowMap, vec4 coords){
  // STEP 1: avgblocker depth
  float avgBlocker_depth = findBlocker(shadowMap,coords.xy,coords.z);   //在这步里我们已经做好了采样，后面就能直接调用数据
  float wLight = 1.0; //光源大小
  float dReceiver = coords.z;

  // STEP 2: penumbra size
  float wPenumbra = wLight * (dReceiver - avgBlocker_depth) / avgBlocker_depth;

  // STEP 3: filtering 就是做PCF，不过加入了wPenumra的影响
  //首先定义变量
  float stride = 10.0;
  float shadowMapSize = 2048.0;
  float visibility1 = 0.0;
  float cur_depth = coords.z;
  float filterRange = stride / shadowMapSize;

  //做采样，前面已经做好了
  //poissonDiskSamples(coords.xy);

  //然后循环比较
  for(int i = 0; i < NUM_SAMPLES; i++){
    float shadow_depth = unpack(texture2D(shadowMap,coords.xy + poissonDisk[i] * filterRange * wPenumbra));
    float res = cur_depth < shadow_depth + EPS ? 1.0 : 0.0;
    visibility1 += res;
  }

  //求平均
  visibility1 /= float(NUM_SAMPLES);
  
  return visibility1;
}

float Bias(float CDepth){
  vec3 lightDir1 = normalize(uLightPos);
  vec3 normal1 = normalize(vNormal);
  float m = 200.0 / 2048.0 / 2.0; // 正交矩阵宽高/shadowmap分辨率/2
  float bias1 = max(m * (1.0-dot(normal1,lightDir1)),m) * CDepth;
  return bias1;
}

float Bias1(){
  vec3 lightDir1 = normalize(uLightPos);
  vec3 normal1 = normalize(vNormal);
  float bias1 = max(0.08 * (1.0-dot(normal1,lightDir1)),0.08);
  return bias1;
}

float useShadowMap1(sampler2D shadowMap, vec4 shadowCoord){
  float mapDepth = unpack(texture2D(shadowMap,shadowCoord.xy));//shadow map中各点的最小深度，unpack将RGBA值转换成[0,1]的float
  float shadingDepth = shadowCoord.z; //当前着色点的深度
  float visibility1 = ((mapDepth + EPS) < shadingDepth) ? 0.0 : 1.0;  
  return visibility1;
}

float useShadowMap(sampler2D shadowMap, vec4 shadowCoord){
  float mapDepth = unpack(texture2D(shadowMap,shadowCoord.xy));//shadow map中各点的最小深度，unpack将RGBA值转换成[0,1]的float
  float shadingDepth = shadowCoord.z; //当前着色点的深度
  //float visibility1 = ((mapDepth + EPS) < shadingDepth) ? 0.0 : 1.0;
  float bias = Bias(1.4);
  float visibility1 = ((mapDepth + EPS) <= (shadingDepth - bias)) ? 0.2 : 0.9;
  return visibility1;
}

vec3 blinnPhong() {
  vec3 color = texture2D(uSampler, vTextureCoord).rgb;
  color = pow(color, vec3(2.2));

  vec3 ambient = 0.05 * color;

  vec3 lightDir = normalize(uLightPos);
  vec3 normal = normalize(vNormal);
  float diff = max(dot(lightDir, normal), 0.0);
  vec3 light_atten_coff =
      uLightIntensity / pow(length(uLightPos - vFragPos), 2.0);
  vec3 diffuse = diff * light_atten_coff * color;

  vec3 viewDir = normalize(uCameraPos - vFragPos);
  vec3 halfDir = normalize((lightDir + viewDir));
  float spec = pow(max(dot(halfDir, normal), 0.0), 32.0);
  vec3 specular = uKs * light_atten_coff * spec;

  vec3 radiance = (ambient + diffuse + specular);
  vec3 phongColor = pow(radiance, vec3(1.0 / 2.2));
  return phongColor;
}

void main(void) {

  float visibility;
  //从裁剪坐标转化为NDC
  vec3 shadowCoord = vPositionFromLight.xyz / vPositionFromLight.w;
  shadowCoord.xyz = (shadowCoord.xyz + 1.0) / 2.0;

  //visibility = useShadowMap(uShadowMap, vec4(shadowCoord, 1.0));
  //visibility = PCF(uShadowMap, vec4(shadowCoord, 1.0));
  visibility = PCSS(uShadowMap, vec4(shadowCoord, 1.0));

  vec3 phongColor = blinnPhong();

  gl_FragColor = vec4(phongColor * visibility, 1.0);
  //gl_FragColor = vec4(phongColor, 1.0);
}