import * as T from 'three';

export function landscape(scene, islands) {
  let seed=731;
  const random=()=>((seed=Math.imul(seed,1664525)+1013904223>>>0)/4294967296);
  const material=color=>new T.MeshStandardMaterial({color,roughness:.92,flatShading:true});
  const batches=new Map();
  const shapes={box:new T.BoxGeometry(), rock:new T.IcosahedronGeometry(1,0), pine:new T.ConeGeometry(1,1,8), post:new T.CylinderGeometry(1,1,1,9)};
  const dummy=new T.Object3D();
  function add(shape,color,x,y,z,w,h,d,ry=0) {
    const key=shape+color;
    if(!batches.has(key)) batches.set(key,{shape,color,matrices:[]});
    dummy.position.set(x,y,z);dummy.scale.set(w,h,d);dummy.rotation.set(0,ry,0);dummy.updateMatrix();
    batches.get(key).matrices.push(dummy.matrix.clone());
  }
  function tree(x,z,size,base=0) {
    add('post',0x664b3b,x,base+.65*size,z,.16*size,1.3*size,.16*size);
    for(let i=0;i<3;i++) add('pine',[0x31584e,0x406b55,0x54805d][i],x,base+(1.3+i*.7)*size,z,(1.04-i*.2)*size,(2.1-i*.28)*size,(1.04-i*.2)*size);
  }
  function shore(x,z,r){
    const sides=46, positions=[],colors=[];
    const rings=[[-3.6,1.17,0x548a90],[-1.7,1.09,0xc6b88f],[-.63,1.02,0xdfcb9e],[-.22,.92,0x999c83],[0,.81,0x93a77f],[0,.01,0xa6b58b]];
    const radii=Array.from({length:sides},(_,i)=>1+.043*Math.sin(i*.81)+random()*.055);
    function vertex(j,i){const angle=i/sides*Math.PI*2;return [x+Math.sin(angle)*r*radii[i]*rings[j][1],rings[j][0],z+Math.cos(angle)*r*radii[i]*rings[j][1]];}
    for(let j=0;j<rings.length-1;j++) for(let i=0;i<sides;i++) {
      const k=(i+1)%sides;
      const a=vertex(j,i),b=vertex(j,k),c=vertex(j+1,i),d=vertex(j+1,k);
      for(const tri of [[a,b,c],[b,d,c]]){
        const color=new T.Color(rings[j][2]).multiplyScalar(.94+.10*random());
        for(const v of tri){positions.push(...v);colors.push(color.r,color.g,color.b);}
      }
    }
    const geo=new T.BufferGeometry();geo.setAttribute('position',new T.Float32BufferAttribute(positions,3));geo.setAttribute('color',new T.Float32BufferAttribute(colors,3));geo.computeVertexNormals();
    const mesh=new T.Mesh(geo,new T.MeshStandardMaterial({vertexColors:true,side:T.DoubleSide,roughness:1}));scene.add(mesh);
  }
  for(const {id,pos,radius} of islands) {
    shore(pos.x,pos.z,radius);
    for(let i=0;i<28;i++){
      const a=random()*Math.PI*2;
      const x=pos.x+Math.sin(a)*radius*.95,z=pos.z+Math.cos(a)*radius*.95, size=.25+random()*.75;
      add('rock',i%3===0?0x8b9690:0x9ea69a,x,-.87,z,size*1.5,size*.75,size,random()*5);
      if(i<10 && (id==='hub'||Math.cos(a)<-.25)) tree(pos.x+Math.sin(a)*radius*.77,pos.z+Math.cos(a)*radius*.77,.65+random());
    }
    if(id==='hub') continue;
    const length=pos.length(),yaw=Math.atan2(pos.x,pos.z);
    for(let t=5;t<length-radius*.67;t+=.56) {
      const x=t*Math.sin(yaw),z=t*Math.cos(yaw);
      add('box',Math.floor(t*10)%3?0xa37850:0xc49b70,x,-.04,z,3,.22,.53,yaw);
      if(Math.round((t-5)/.56)%6===0) for(const side of [-1,1]) {
        const off=1.48*side;
        add('post',0x685347,x+Math.cos(yaw)*off,-.5,z-Math.sin(yaw)*off,.13,3.9,.13);
      }
    }
    for(const side of [-1,1]) {
      const end=length-radius*.67,mid=(5+end)/2;
      add('box',0x957859,mid*Math.sin(yaw)+side*1.48*Math.cos(yaw),1.08,mid*Math.cos(yaw)-side*1.48*Math.sin(yaw),.12,.13,end-5,yaw);
    }
  }
  for(let i=0;i<75;i++){
    const a=i/75*Math.PI*2,r=145+random()*75,h=9+random()*24;
    add('rock',i%3?0x728d94:0x889e9d,Math.sin(a)*r,-4,Math.cos(a)*r,16+random()*22,h,23,random()*6);
    if(h>26)add('rock',0xc3d0c8,Math.sin(a)*r,h*.54-4,Math.cos(a)*r,13,h*.39,14);
  }
  for(const {shape,color,matrices} of batches.values()){
    const mesh=new T.InstancedMesh(shapes[shape],material(color),matrices.length);
    matrices.forEach((m,i)=>mesh.setMatrixAt(i,m));
    scene.add(mesh);
  }
  const sea=new T.Mesh(new T.PlaneGeometry(620,620),new T.ShaderMaterial({
    uniforms:{time:{value:0}},vertexShader:`varying vec3 p;void main(){p=(modelMatrix*vec4(position,1.)).xyz;gl_Position=projectionMatrix*viewMatrix*vec4(p,1.);}`,
    fragmentShader:`varying vec3 p;uniform float time;
    void main(){vec2 q=p.xz;float t=time*.68;
      float w=sin(q.x*1.5+q.y*.51+t+sin(q.y*.7+t*.6)*1.25)*.5+.5;
      float ripple=pow(w,14.)*(.5+.5*sin(q.y*.79-q.x*.18+t*.7));
      float band=sin(q.y*.08+q.x*.03)*.5+.5;
      vec3 col=mix(vec3(.055,.34,.40),vec3(.16,.54,.55),band);
      col+=vec3(.22,.28,.24)*ripple;
      float dist=length(cameraPosition-p);col=mix(col,vec3(.69,.82,.83),smoothstep(52.,220.,dist));
      gl_FragColor=vec4(col,1.);
      #include <colorspace_fragment>
    }`.replace('#include <colorspace_fragment>', '\n#include <colorspace_fragment>\n')
  }));
  sea.rotation.x=-Math.PI/2;sea.position.y=-1.6;scene.add(sea);
  const sky=new T.Mesh(new T.SphereGeometry(285,32,14),new T.ShaderMaterial({side:T.BackSide,depthWrite:false,
    vertexShader:'varying vec3 dir;void main(){dir=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}',
    fragmentShader:`varying vec3 dir;void main(){vec3 d=normalize(dir);vec3 c=mix(vec3(.77,.87,.87),vec3(.14,.43,.65),smoothstep(.01,.87,d.y));float glow=max(0.,dot(d,normalize(vec3(-.55,.46,-.66))));c+=vec3(.8,.64,.37)*(pow(glow,100.)*.45+smoothstep(.999,.9997,glow));gl_FragColor=vec4(c,1.);
#include <colorspace_fragment>
}`}));
  scene.add(sky);
  return t=>{sea.material.uniforms.time.value=t;};
}
