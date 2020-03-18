// based on iq's sculpture https://www.shadertoy.com/view/XtjSDK
Shader "Unlit/Sculpture"
{
    Properties
    {
        _MainTex ("Texture", 2D) = "white" {}
    }
    SubShader
    {
        Tags { "RenderType"="Opaque" }
        LOD 100

        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            // make fog work
            #pragma multi_compile_fog

            #include "UnityCG.cginc"

            struct appdata
            {
                float4 vertex : POSITION;
                float2 uv : TEXCOORD0;
            };

            struct v2f
            {
                float2 uv : TEXCOORD0;
                UNITY_FOG_COORDS(1)
                float4 vertex : SV_POSITION;
            };

            sampler2D _MainTex;
            float4 _MainTex_ST;

            v2f vert (appdata v)
            {
                v2f o;
                o.vertex = UnityObjectToClipPos(v.vertex);
                o.uv = TRANSFORM_TEX(v.uv, _MainTex);
                UNITY_TRANSFER_FOG(o,o.vertex);
                return o;
            }

#define AA 1

			float hash1(float n) {
				return frac(sin(n)*43758.5453123);
			}

			float hash1(in float2 f) {
				return frac(sin(f.x + 131.1*f.y)*43758.5453123);
			}

			static const float PI = 3.1415926535897932384626433832795;
			static const float PHI = 1.6180339887498948482045868343656;

			float3 forwardSF(float i, float n) {
				float phi = 2.0*PI*frac(i / PHI);
				float zi = 1.0 - (2.0*i + 1.0) / n;
				float sinTheta = sqrt(1.0 - zi * zi);
				return float3(cos(phi)*sinTheta, sin(phi)*sinTheta, zi);
			}

			float4 grow = float4(1.0, 1.0, 1.0, 1.0);

			float3 mapP(float3 p) {
				p.xyz += 0.800*sin(2.0*p.yzx)*grow.x;
				p.xyz += cos(_Time.y)*0.500*sin(4.0*p.yzx)*grow.y;
				p.xyz += sin(_Time.y)*0.250*sin(8.0*p.yzx)*grow.z;
				//p.xyz += abs(cos(_Time.y*0.5))*1.750*sin(16.0*p.yzx)*grow.w;
				p.xyz += cos(_Time.y*0.5)*1.750*sin(16.0*p.yzx)*grow.w;
				//p.xyz += 1.250*sin(4.0*p.yzx)*grow.w;
				//p.xyz += 1.050*sin(8.0*p.yzx)*grow.w;
				return p;
			}

			float map(float3 q) {
				float3 p = mapP(q);
				float d = length(p) - 1.5;
				return d * 0.05;
			}

			float intersect(in float3 ro, in float3 rd) {
				static const float maxd = 24.0;
				
				float precis = 0.001;
				float h = 1.0;
				float t = 1.0;
				for (int i = 0; i < 1256; i++)
				{
					if ((h < precis) || (t > maxd))
					{
						break;
					}
					h = map(ro + rd * t);
					t += h;
				}

				if (t > maxd)
				{
					t = -1.0;
				}
				return t;
			}

			float3 calcNormal(in float3 pos) {
				float3 eps = float3(0.005, 0.0, 0.0);
				return normalize(float3(
					map(pos + eps.xyy) - map(pos - eps.xyy),
					map(pos + eps.yxy) - map(pos - eps.yxy),
					map(pos + eps.yyx) - map(pos - eps.yyx)
					));
			}

			float calcAO(in float3 pos, in float3 nor, in float2 pix) {
				float ao = 0.0;
				for (int i = 0; i < 64; i++)
				{
					float3 ap = forwardSF(float(i), 64.0);
					ap *= sign(dot(ap, nor)) * hash1(float(i));
					ao += clamp(map(pos + nor * 0.05 + ap * 1.0) * 32.0, 0.0, 1.0);
				}
				ao / 64.0;

				return clamp(ao*ao, 0.0, 1.0);
			}

			float calcAO2(in float3 pos, in float3 nor, in float2 pix) {
				float ao = 0.0;
				for (int i = 0; i < 32; i++)
				{
					float3 ap = forwardSF(float(i), 32.0);
					ap *= sign(dot(ap, nor)) * hash1(float(i));
					ao += clamp(map(pos + nor * 0.05 + ap * 0.2)*100.0, 0.0, 1.0);
				}
				ao /= 32.0;

				return clamp(ao, 0.0, 1.0);
			}

            fixed4 frag (v2f i) : SV_Target
            {
#define ZERO (min(int(_Time.y), 0))
				float3 tot = float3(0.0, 0.0, 0.0);
				// AA is 1...
				for (int m = ZERO; m < AA; m++)
				{
					for (int n = 0; n < AA; n++)
					{
						// pixel coordinates
						float2 o = float2(float(m), float(n)) / float(AA) - 0.5;
						float2 p = (2.0*(i.uv.xy*_ScreenParams.xy + o) - _ScreenParams.xy) / _ScreenParams.y;
						float2 q = (i.uv.xy*_ScreenParams.xy + o) / _ScreenParams.xy;

						grow = smoothstep(0.0, 1.0, (_Time.y - float4(0.0, 1.0, 2.0, 3.0)) / 3.0);

						// camera
						float an = 1.1 + 0.5*(_Time.y - 10.0);
						//float3 ro = float3(8.5*sin(an), -1.0*cos(an), -8.5*cos(an));
						float3 ro = float3(-6.5*sin(an), -6.5*cos(an), 7.0);
						//float3 ro = float3(8.5*sin(an), -8.5*cos(an), 1.0);
						//float3 ro = float3((6.0+1.0*sin(_Time.y))*sin(an), (6.0 + 1.0*sin(_Time.y))*cos(an), 1.0);
						float3 ta = float3(0.0, 0.2, 0.0);
						//float3 ta = float3(0.0, 0.0, 0.0);

						// camera matrix
						float3 ww = normalize(ta - ro);
						float3 uu = normalize(cross(ww, float3(0.0, 1.0, 0.0)));
						float3 vv = normalize(cross(uu, ww));

						// create view ray
						float3 rd = normalize(p.x*uu + p.y*vv + 1.5*ww);

						//
						// render
						//
						//float3 col = float3(0.07, 0.07, 0.07)*clamp(1.0 - length(q - 0.5), 0.0, 1.0);
						float3 col = float3(0.11, 0.11, 0.11)*clamp(1.0 - length(q - 0.5), 0.0, 1.0);

						// raymarch
						float t = intersect(ro, rd);

						if (t > 0.0)
						{
							float3 pos = ro + t * rd;
							float3 nor = calcNormal(pos);
							float3 ref = reflect(rd, nor);
							float3 sor = nor;

							float3 q = mapP(pos);
							float occ = calcAO(pos, nor, i.uv.xy * _ScreenParams.xy); occ = occ * occ;

							// materials
							col = float3(0.04, 0.04, 0.04);
							float ar = clamp(1.0 - 0.7*length(q - pos), 0.0, 1.0);
							col = lerp(col, float3(0.001, 0.001, 0.001), ar);
							col *= 60.;
							//col *= lerp(float3(.4, .4, .6), .5*float3(1.0+abs(cos(_Time.y*0.5))*2.0, 1. + abs(cos(_Time.y*0.5))*2.0, 1. + abs(cos(_Time.y*0.5))*2.0), occ);
							float occ2 = calcAO2(pos, nor, i.uv.xy * _ScreenParams.xy);

							//col *= 1.0*lerp(float3(5.5*abs(cos	(_Time.y)) + 10., 10.5*abs(sin(_Time.y))+3.5, 5.5*abs(cos(_Time.y)) + 1.), float3(1., 3., 13.), occ2*occ2*occ2);
							//col.b *= 0.5;
							col *= .2;
							//col *= 1.0*lerp(float3(1., 1, 1.), float3(.0, .0, .0), occ2);
							float ks = 0.0;

							// lighting
							float sky = 0.5 + 0.5*nor.y;
							float fre = clamp(1.0 + dot(nor, rd), 0.0, 1.0);
							float spe = pow(max(dot(-rd, nor), 0.0), 8.0);
							// lights
							float3 lin = float3(0.0, 0.0, 0.0);
							lin += 3.0*float3(0.7, 0.8, 1.0)*sky*occ;
							//lin += 1.0*fre*float3(1.2, 0.7, 0.6)*(0.1 + 0.9*occ);
							lin += 1.0*fre*float3(0.8, 0.7, 0.9)*(0.1 + 0.9*occ);
							col += 0.3*ks*4.0*float3(0.7, 0.8, 1.0)*smoothstep(0.0, 0.2, ref.y)*(0.05 + 0.95*pow(fre, 5.0))*(0.5 + 0.5*nor.y)*occ;
							col += 4.0*ks*1.5*spe*occ*col.x;
							col += 2.0*ks*1.0*pow(spe, 8.0)*occ*col.x;
							col = col * lin;
						}

						col = pow(col, float3(0.4545, 0.4545, 0.4545));
						tot += col;
					
					
					}
				}
				
				tot /= float(AA*AA);

				//tot = pow(tot, float3(1.0, 1.0, 1.4)) + float3(0.0, 0.02, 0.14);

				tot += (1.0 / 255.0) * hash1(i.uv.xy*_ScreenParams.xy);

				return float4(tot, 1.0);
            }
            ENDCG
        }
    }
}
