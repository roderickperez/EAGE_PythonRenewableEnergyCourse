var Vs=[{id:"python",name:"Python Island",x:0,z:8,color:"#3676a5",quiz:"generalPython",description:"In the Python Course, run a notebook, store measurements in variables and lists, and use loops and functions to repeat a calculation. NumPy, pandas and Matplotlib take you from individual readings to arrays, tables and plots.",task:"Calculate energy from a short power series before crossing to the energy hub.",question:"In Python, what do 7 / 2 and 7 // 2 return?",options:["3 and 3","3.5 and 3","3.5 and 4"],correct:1,why:"True division gives 3.5. Floor division rounds the quotient down to 3."},{id:"hydro",name:"Hydropower",x:-8,z:3,color:"#237fb6",quiz:"hydroelectricEnergy",description:"Trace water from the reservoir to the turbine. Use discharge, net head and efficiency to estimate electrical power, then integrate power over time to get energy.",task:"Change the flow in the plant lab and watch the turbine and power readout.",question:"Which pair gives the units of electrical power and delivered energy, in that order?",options:["MWh and MW","MW and MW","MW and MWh"],correct:2,why:"A plant operating at 1 MW for 1 hour delivers 1 MWh."},{id:"solar",name:"Solar",x:-5,z:-6,color:"#d78b23",quiz:"solarEnergy",description:"Follow sunlight from the receiving plane to the panel and inverter. Calculate DC power from irradiance, module area and efficiency, then account for temperature and conversion losses.",task:"Compare panel output under two irradiance settings.",question:"At 780 W/m\xB2, how much ideal DC power does a 2.1 m\xB2 panel with 21% efficiency produce?",options:["343.98 W","1,638 W","78 W"],correct:0,why:"780 \xD7 2.1 \xD7 0.21 = 343.98 W before system losses."},{id:"wind",name:"Wind",x:4,z:-7,color:"#318976",quiz:"windEnergy",description:"Follow kinetic energy through the rotor and drivetrain. Apply a turbine power curve to a wind-speed series to estimate annual energy and capacity factor.",task:"Vary the wind speed and compare the resource with turbine output.",question:"Wind speed doubles from 5 to 10 m/s. With fixed density and rotor area, how does available wind power change?",options:["It doubles","It increases eightfold","It quadruples"],correct:1,why:"Available power scales with speed cubed, so the factor is 2\xB3 = 8."},{id:"geothermal",name:"Geothermal",x:8,z:2,color:"#be6547",quiz:"geothermalEnergy",description:"Trace production and reinjection through a geothermal reservoir. Use mass flow and the temperature drop to estimate extracted heat, then distinguish thermal output from electrical generation.",task:"Follow the fluid circuit in the lab and identify where heat leaves the system.",question:"Which unit describes the geothermal gradient?",options:["W/m\xB2","MWh","\xB0C/km"],correct:2,why:"Gradient measures temperature change per unit depth. W/m\xB2 measures heat flux."}];/**
 * @license
 * Copyright 2010-2026 Three.js Authors
 * SPDX-License-Identifier: MIT
 */var Mi={LEFT:0,MIDDLE:1,RIGHT:2,ROTATE:0,DOLLY:1,PAN:2},Si={ROTATE:0,PAN:1,DOLLY_PAN:2,DOLLY_ROTATE:3},Zh=0,$l=1,Jh=2;var Dr=1,Zl=2,Ls=3,wi=0,We=1,Ee=2,Xi=0,Ds=1,Jl=2,Kl=3,jl=4,Kh=5;var qi=100,jh=101,Qh=102,tu=103,eu=104,nu=200,iu=201,su=202,ru=203,Ql=204,tc=205,au=206,ou=207,lu=208,cu=209,hu=210,uu=211,du=212,fu=213,pu=214,Ra=0,Pa=1,Ia=2,ps=3,La=4,Da=5,Na=6,Ua=7,ec=0,mu=1,gu=2,Bn=0,nc=1,ic=2,sc=3,Pn=4,rc=5,ac=6,oc=7;var lc=300,Yi=301,Ns=302,ao=303,oo=304,ms=1e3,mn=1001,gs=1002,Ge=1003,xu=1004;var Nr=1005;var je=1006,lo=1007;var si=1008;var nn=1009,cc=1010,hc=1011,Us=1012,co=1013,In=1014,xn=1015,Ln=1016,ho=1017,uo=1018,$i=1020,uc=35902,dc=35899,fc=1021,pc=1022,Dn=1023,Kn=1026,ri=1027,fo=1028,po=1029,Ti=1030,mo=1031;var go=1033,Ur=33776,Fr=33777,Or=33778,Br=33779,xo=35840,_o=35841,yo=35842,vo=35843,bo=36196,Mo=37492,So=37496,wo=37488,To=37489,kr=37490,Ao=37491,Eo=37808,Co=37809,Ro=37810,Po=37811,Io=37812,Lo=37813,Do=37814,No=37815,Uo=37816,Fo=37817,Oo=37818,Bo=37819,ko=37820,zo=37821,Go=36492,Vo=36494,Ho=36495,Wo=36283,Xo=36284,zr=36285,qo=36286;var nr=2300,Fa=2301,Ea=2302,Ul=2303,Fl=2400,Ol=2401,Bl=2402;var _u=3200;var Yo=0,yu=1,ai="",He="srgb",ir="srgb-linear",sr="linear",ge="srgb";var Ca=7680;var vu=519,bu=512,Mu=513,Su=514,mc=515,wu=516,Tu=517,gc=518,xc=519,Au=35044;var _c="300 es",gn=2e3,xs=2001;function af(i){for(let t=i.length-1;t>=0;--t)if(i[t]>=65535)return!0;return!1}function of(i){return ArrayBuffer.isView(i)&&!(i instanceof DataView)}function rr(i){return document.createElementNS("http://www.w3.org/1999/xhtml",i)}function Eu(){let i=rr("canvas");return i.style.display="block",i}var ch={},_s=null;function yc(...i){let t="THREE."+i.shift();_s?_s("log",t,...i):console.log(t,...i)}function Cu(i){let t=i[0];if(typeof t=="string"&&t.startsWith("TSL:")){let e=i[1];e&&e.isStackTrace?i[0]+=" "+e.getLocation():i[1]='Stack trace not available. Enable "THREE.Node.captureStackTrace" to capture stack traces.'}return i}function qt(...i){i=Cu(i);let t="THREE."+i.shift();if(_s)_s("warn",t,...i);else{let e=i[0];e&&e.isStackTrace?console.warn(e.getError(t)):console.warn(t,...i)}}function Zt(...i){i=Cu(i);let t="THREE."+i.shift();if(_s)_s("error",t,...i);else{let e=i[0];e&&e.isStackTrace?console.error(e.getError(t)):console.error(t,...i)}}function Ks(...i){let t=i.join(" ");t in ch||(ch[t]=!0,qt(...i))}function Ru(i,t,e){return new Promise(function(n,s){function r(){switch(i.clientWaitSync(t,i.SYNC_FLUSH_COMMANDS_BIT,0)){case i.WAIT_FAILED:s();break;case i.TIMEOUT_EXPIRED:setTimeout(r,e);break;default:n()}}setTimeout(r,e)})}var Pu={[Ra]:Ra,[Ia]:Na,[La]:La,[ps]:Da,[Pa]:Pa,[Na]:Ia,[Ua]:Ua,[Da]:ps},Sn=class{addEventListener(t,e){this._listeners===void 0&&(this._listeners={});let n=this._listeners;n[t]===void 0&&(n[t]=[]),n[t].indexOf(e)===-1&&n[t].push(e)}hasEventListener(t,e){let n=this._listeners;return n===void 0?!1:n[t]!==void 0&&n[t].indexOf(e)!==-1}removeEventListener(t,e){let n=this._listeners;if(n===void 0)return;let s=n[t];if(s!==void 0){let r=s.indexOf(e);r!==-1&&s.splice(r,1)}}dispatchEvent(t){let e=this._listeners;if(e===void 0)return;let n=e[t.type];if(n!==void 0){t.target=this;let s=n.slice(0);for(let r=0,a=s.length;r<a;r++)s[r].call(this,t);t.target=null}}},Ye=["00","01","02","03","04","05","06","07","08","09","0a","0b","0c","0d","0e","0f","10","11","12","13","14","15","16","17","18","19","1a","1b","1c","1d","1e","1f","20","21","22","23","24","25","26","27","28","29","2a","2b","2c","2d","2e","2f","30","31","32","33","34","35","36","37","38","39","3a","3b","3c","3d","3e","3f","40","41","42","43","44","45","46","47","48","49","4a","4b","4c","4d","4e","4f","50","51","52","53","54","55","56","57","58","59","5a","5b","5c","5d","5e","5f","60","61","62","63","64","65","66","67","68","69","6a","6b","6c","6d","6e","6f","70","71","72","73","74","75","76","77","78","79","7a","7b","7c","7d","7e","7f","80","81","82","83","84","85","86","87","88","89","8a","8b","8c","8d","8e","8f","90","91","92","93","94","95","96","97","98","99","9a","9b","9c","9d","9e","9f","a0","a1","a2","a3","a4","a5","a6","a7","a8","a9","aa","ab","ac","ad","ae","af","b0","b1","b2","b3","b4","b5","b6","b7","b8","b9","ba","bb","bc","bd","be","bf","c0","c1","c2","c3","c4","c5","c6","c7","c8","c9","ca","cb","cc","cd","ce","cf","d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","da","db","dc","dd","de","df","e0","e1","e2","e3","e4","e5","e6","e7","e8","e9","ea","eb","ec","ed","ee","ef","f0","f1","f2","f3","f4","f5","f6","f7","f8","f9","fa","fb","fc","fd","fe","ff"],hh=1234567,js=Math.PI/180,ys=180/Math.PI;function Zi(){let i=Math.random()*4294967295|0,t=Math.random()*4294967295|0,e=Math.random()*4294967295|0,n=Math.random()*4294967295|0;return(Ye[i&255]+Ye[i>>8&255]+Ye[i>>16&255]+Ye[i>>24&255]+"-"+Ye[t&255]+Ye[t>>8&255]+"-"+Ye[t>>16&15|64]+Ye[t>>24&255]+"-"+Ye[e&63|128]+Ye[e>>8&255]+"-"+Ye[e>>16&255]+Ye[e>>24&255]+Ye[n&255]+Ye[n>>8&255]+Ye[n>>16&255]+Ye[n>>24&255]).toLowerCase()}function ne(i,t,e){return Math.max(t,Math.min(e,i))}function vc(i,t){return(i%t+t)%t}function lf(i,t,e,n,s){return n+(i-t)*(s-n)/(e-t)}function cf(i,t,e){return i!==t?(e-i)/(t-i):0}function Qs(i,t,e){return(1-e)*i+e*t}function hf(i,t,e,n){return Qs(i,t,1-Math.exp(-e*n))}function uf(i,t=1){return t-Math.abs(vc(i,t*2)-t)}function df(i,t,e){return i<=t?0:i>=e?1:(i=(i-t)/(e-t),i*i*(3-2*i))}function ff(i,t,e){return i<=t?0:i>=e?1:(i=(i-t)/(e-t),i*i*i*(i*(i*6-15)+10))}function pf(i,t){return i+Math.floor(Math.random()*(t-i+1))}function mf(i,t){return i+Math.random()*(t-i)}function gf(i){return i*(.5-Math.random())}function xf(i){i!==void 0&&(hh=i);let t=hh+=1831565813;return t=Math.imul(t^t>>>15,t|1),t^=t+Math.imul(t^t>>>7,t|61),((t^t>>>14)>>>0)/4294967296}function _f(i){return i*js}function yf(i){return i*ys}function vf(i){return i>0&&Number.isInteger(i)&&2**Math.round(Math.log2(i))===i}function bf(i){return Math.pow(2,Math.ceil(Math.log(i)/Math.LN2))}function bc(i){return Math.pow(2,Math.floor(Math.log(i)/Math.LN2))}function Mf(i,t,e,n,s){let r=Math.cos,a=Math.sin,o=r(e/2),c=a(e/2),l=r((t+n)/2),u=a((t+n)/2),h=r((t-n)/2),d=a((t-n)/2),f=r((n-t)/2),m=a((n-t)/2);switch(s){case"XYX":i.set(o*u,c*h,c*d,o*l);break;case"YZY":i.set(c*d,o*u,c*h,o*l);break;case"ZXZ":i.set(c*h,c*d,o*u,o*l);break;case"XZX":i.set(o*u,c*m,c*f,o*l);break;case"YXY":i.set(c*f,o*u,c*m,o*l);break;case"ZYZ":i.set(c*m,c*f,o*u,o*l);break;default:qt("MathUtils: .setQuaternionFromProperEuler() encountered an unknown order: "+s)}}function Mc(i){switch(i.constructor){case Uint32Array:return 4294967295;case Uint16Array:return 65535;case Uint8Array:case Uint8ClampedArray:return 255;case Int32Array:return 2147483647;case Int16Array:return 32767;case Int8Array:return 127;default:return}}function Iu(i,t){if(t.constructor===Float32Array)return i;let e=Mc(t);if(e===void 0)throw new Error("THREE.MathUtils: Invalid component type.");return Math.max(i/e,-1)}function Lu(i,t){if(t.constructor===Float32Array)return i;let e=Mc(t);if(e===void 0)throw new Error("THREE.MathUtils: Invalid component type.");return Math.round(i*e)}var oi={DEG2RAD:js,RAD2DEG:ys,generateUUID:Zi,clamp:ne,euclideanModulo:vc,mapLinear:lf,inverseLerp:cf,lerp:Qs,damp:hf,pingpong:uf,smoothstep:df,smootherstep:ff,randInt:pf,randFloat:mf,randFloatSpread:gf,seededRandom:xf,degToRad:_f,radToDeg:yf,isPowerOfTwo:vf,ceilPowerOfTwo:bf,floorPowerOfTwo:bc,setQuaternionFromProperEuler:Mf,normalize:Lu,denormalize:Iu},ft=class{static{this.prototype.isVector2=!0}constructor(t=0,e=0){this.x=t,this.y=e}get width(){return this.x}set width(t){this.x=t}get height(){return this.y}set height(t){this.y=t}set(t,e){return this.x=t,this.y=e,this}setScalar(t){return this.x=t,this.y=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;default:throw new Error("THREE.Vector2: index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;default:throw new Error("THREE.Vector2: index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y)}copy(t){return this.x=t.x,this.y=t.y,this}add(t){return this.x+=t.x,this.y+=t.y,this}addScalar(t){return this.x+=t,this.y+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this}subScalar(t){return this.x-=t,this.y-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this}multiply(t){return this.x*=t.x,this.y*=t.y,this}multiplyScalar(t){return this.x*=t,this.y*=t,this}divide(t){return this.x/=t.x,this.y/=t.y,this}divideScalar(t){return this.multiplyScalar(1/t)}applyMatrix3(t){let e=this.x,n=this.y,s=t.elements;return this.x=s[0]*e+s[3]*n+s[6],this.y=s[1]*e+s[4]*n+s[7],this}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this}clamp(t,e){return this.x=ne(this.x,t.x,e.x),this.y=ne(this.y,t.y,e.y),this}clampScalar(t,e){return this.x=ne(this.x,t,e),this.y=ne(this.y,t,e),this}clampLength(t,e){let n=this.length();return this.divideScalar(n||1).multiplyScalar(ne(n,t,e))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this}negate(){return this.x=-this.x,this.y=-this.y,this}dot(t){return this.x*t.x+this.y*t.y}cross(t){return this.x*t.y-this.y*t.x}lengthSq(){return this.x*this.x+this.y*this.y}length(){return Math.sqrt(this.x*this.x+this.y*this.y)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)}normalize(){return this.divideScalar(this.length()||1)}angle(){return Math.atan2(-this.y,-this.x)+Math.PI}angleTo(t){let e=Math.sqrt(this.lengthSq()*t.lengthSq());if(e===0)return Math.PI/2;let n=this.dot(t)/e;return Math.acos(ne(n,-1,1))}distanceTo(t){return Math.sqrt(this.distanceToSquared(t))}distanceToSquared(t){let e=this.x-t.x,n=this.y-t.y;return e*e+n*n}manhattanDistanceTo(t){return Math.abs(this.x-t.x)+Math.abs(this.y-t.y)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this}equals(t){return t.x===this.x&&t.y===this.y}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this}rotateAround(t,e){let n=Math.cos(e),s=Math.sin(e),r=this.x-t.x,a=this.y-t.y;return this.x=r*n-a*s+t.x,this.y=r*s+a*n+t.y,this}random(){return this.x=Math.random(),this.y=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y}},ln=class{constructor(t=0,e=0,n=0,s=1){this.isQuaternion=!0,this._x=t,this._y=e,this._z=n,this._w=s}static slerpFlat(t,e,n,s,r,a,o){let c=n[s+0],l=n[s+1],u=n[s+2],h=n[s+3],d=r[a+0],f=r[a+1],m=r[a+2],y=r[a+3];if(h!==y||c!==d||l!==f||u!==m){let g=c*d+l*f+u*m+h*y;g<0&&(d=-d,f=-f,m=-m,y=-y,g=-g);let p=1-o;if(g<.9995){let v=Math.acos(g),b=Math.sin(v);p=Math.sin(p*v)/b,o=Math.sin(o*v)/b,c=c*p+d*o,l=l*p+f*o,u=u*p+m*o,h=h*p+y*o}else{c=c*p+d*o,l=l*p+f*o,u=u*p+m*o,h=h*p+y*o;let v=1/Math.sqrt(c*c+l*l+u*u+h*h);c*=v,l*=v,u*=v,h*=v}}t[e]=c,t[e+1]=l,t[e+2]=u,t[e+3]=h}static multiplyQuaternionsFlat(t,e,n,s,r,a){let o=n[s],c=n[s+1],l=n[s+2],u=n[s+3],h=r[a],d=r[a+1],f=r[a+2],m=r[a+3];return t[e]=o*m+u*h+c*f-l*d,t[e+1]=c*m+u*d+l*h-o*f,t[e+2]=l*m+u*f+o*d-c*h,t[e+3]=u*m-o*h-c*d-l*f,t}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get w(){return this._w}set w(t){this._w=t,this._onChangeCallback()}set(t,e,n,s){return this._x=t,this._y=e,this._z=n,this._w=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._w)}copy(t){return this._x=t.x,this._y=t.y,this._z=t.z,this._w=t.w,this._onChangeCallback(),this}setFromEuler(t,e=!0){let n=t._x,s=t._y,r=t._z,a=t._order,o=Math.cos,c=Math.sin,l=o(n/2),u=o(s/2),h=o(r/2),d=c(n/2),f=c(s/2),m=c(r/2);switch(a){case"XYZ":this._x=d*u*h+l*f*m,this._y=l*f*h-d*u*m,this._z=l*u*m+d*f*h,this._w=l*u*h-d*f*m;break;case"YXZ":this._x=d*u*h+l*f*m,this._y=l*f*h-d*u*m,this._z=l*u*m-d*f*h,this._w=l*u*h+d*f*m;break;case"ZXY":this._x=d*u*h-l*f*m,this._y=l*f*h+d*u*m,this._z=l*u*m+d*f*h,this._w=l*u*h-d*f*m;break;case"ZYX":this._x=d*u*h-l*f*m,this._y=l*f*h+d*u*m,this._z=l*u*m-d*f*h,this._w=l*u*h+d*f*m;break;case"YZX":this._x=d*u*h+l*f*m,this._y=l*f*h+d*u*m,this._z=l*u*m-d*f*h,this._w=l*u*h-d*f*m;break;case"XZY":this._x=d*u*h-l*f*m,this._y=l*f*h-d*u*m,this._z=l*u*m+d*f*h,this._w=l*u*h+d*f*m;break;default:qt("Quaternion: .setFromEuler() encountered an unknown order: "+a)}return e===!0&&this._onChangeCallback(),this}setFromAxisAngle(t,e){let n=e/2,s=Math.sin(n);return this._x=t.x*s,this._y=t.y*s,this._z=t.z*s,this._w=Math.cos(n),this._onChangeCallback(),this}setFromRotationMatrix(t){let e=t.elements,n=e[0],s=e[4],r=e[8],a=e[1],o=e[5],c=e[9],l=e[2],u=e[6],h=e[10],d=n+o+h;if(d>0){let f=.5/Math.sqrt(d+1);this._w=.25/f,this._x=(u-c)*f,this._y=(r-l)*f,this._z=(a-s)*f}else if(n>o&&n>h){let f=2*Math.sqrt(1+n-o-h);this._w=(u-c)/f,this._x=.25*f,this._y=(s+a)/f,this._z=(r+l)/f}else if(o>h){let f=2*Math.sqrt(1+o-n-h);this._w=(r-l)/f,this._x=(s+a)/f,this._y=.25*f,this._z=(c+u)/f}else{let f=2*Math.sqrt(1+h-n-o);this._w=(a-s)/f,this._x=(r+l)/f,this._y=(c+u)/f,this._z=.25*f}return this._onChangeCallback(),this}setFromUnitVectors(t,e){let n=t.dot(e)+1;return n<1e-8?(n=0,Math.abs(t.x)>Math.abs(t.z)?(this._x=-t.y,this._y=t.x,this._z=0,this._w=n):(this._x=0,this._y=-t.z,this._z=t.y,this._w=n)):(this._x=t.y*e.z-t.z*e.y,this._y=t.z*e.x-t.x*e.z,this._z=t.x*e.y-t.y*e.x,this._w=n),this.normalize()}angleTo(t){return 2*Math.acos(Math.abs(ne(this.dot(t),-1,1)))}rotateTowards(t,e){let n=this.angleTo(t);if(n===0)return this;let s=Math.min(1,e/n);return this.slerp(t,s),this}identity(){return this.set(0,0,0,1)}invert(){return this.conjugate()}conjugate(){return this._x*=-1,this._y*=-1,this._z*=-1,this._onChangeCallback(),this}dot(t){return this._x*t._x+this._y*t._y+this._z*t._z+this._w*t._w}lengthSq(){return this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w}length(){return Math.sqrt(this._x*this._x+this._y*this._y+this._z*this._z+this._w*this._w)}normalize(){let t=this.length();return t===0?(this._x=0,this._y=0,this._z=0,this._w=1):(t=1/t,this._x=this._x*t,this._y=this._y*t,this._z=this._z*t,this._w=this._w*t),this._onChangeCallback(),this}multiply(t){return this.multiplyQuaternions(this,t)}premultiply(t){return this.multiplyQuaternions(t,this)}multiplyQuaternions(t,e){let n=t._x,s=t._y,r=t._z,a=t._w,o=e._x,c=e._y,l=e._z,u=e._w;return this._x=n*u+a*o+s*l-r*c,this._y=s*u+a*c+r*o-n*l,this._z=r*u+a*l+n*c-s*o,this._w=a*u-n*o-s*c-r*l,this._onChangeCallback(),this}slerp(t,e){let n=t._x,s=t._y,r=t._z,a=t._w,o=this.dot(t);o<0&&(n=-n,s=-s,r=-r,a=-a,o=-o);let c=1-e;if(o<.9995){let l=Math.acos(o),u=Math.sin(l);c=Math.sin(c*l)/u,e=Math.sin(e*l)/u,this._x=this._x*c+n*e,this._y=this._y*c+s*e,this._z=this._z*c+r*e,this._w=this._w*c+a*e,this._onChangeCallback()}else this._x=this._x*c+n*e,this._y=this._y*c+s*e,this._z=this._z*c+r*e,this._w=this._w*c+a*e,this.normalize();return this}slerpQuaternions(t,e,n){return this.copy(t).slerp(e,n)}random(){let t=2*Math.PI*Math.random(),e=2*Math.PI*Math.random(),n=Math.random(),s=Math.sqrt(1-n),r=Math.sqrt(n);return this.set(s*Math.sin(t),s*Math.cos(t),r*Math.sin(e),r*Math.cos(e))}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._w===this._w}fromArray(t,e=0){return this._x=t[e],this._y=t[e+1],this._z=t[e+2],this._w=t[e+3],this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._w,t}fromBufferAttribute(t,e){return this._x=t.getX(e),this._y=t.getY(e),this._z=t.getZ(e),this._w=t.getW(e),this._onChangeCallback(),this}toJSON(){return this.toArray()}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._w}},F=class{static{this.prototype.isVector3=!0}constructor(t=0,e=0,n=0){this.x=t,this.y=e,this.z=n}set(t,e,n){return n===void 0&&(n=this.z),this.x=t,this.y=e,this.z=n,this}setScalar(t){return this.x=t,this.y=t,this.z=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setZ(t){return this.z=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;case 2:this.z=e;break;default:throw new Error("THREE.Vector3: index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;case 2:return this.z;default:throw new Error("THREE.Vector3: index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y,this.z)}copy(t){return this.x=t.x,this.y=t.y,this.z=t.z,this}add(t){return this.x+=t.x,this.y+=t.y,this.z+=t.z,this}addScalar(t){return this.x+=t,this.y+=t,this.z+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this.z=t.z+e.z,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this.z+=t.z*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this.z-=t.z,this}subScalar(t){return this.x-=t,this.y-=t,this.z-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this.z=t.z-e.z,this}multiply(t){return this.x*=t.x,this.y*=t.y,this.z*=t.z,this}multiplyScalar(t){return this.x*=t,this.y*=t,this.z*=t,this}multiplyVectors(t,e){return this.x=t.x*e.x,this.y=t.y*e.y,this.z=t.z*e.z,this}applyEuler(t){return this.applyQuaternion(uh.setFromEuler(t))}applyAxisAngle(t,e){return this.applyQuaternion(uh.setFromAxisAngle(t,e))}applyMatrix3(t){let e=this.x,n=this.y,s=this.z,r=t.elements;return this.x=r[0]*e+r[3]*n+r[6]*s,this.y=r[1]*e+r[4]*n+r[7]*s,this.z=r[2]*e+r[5]*n+r[8]*s,this}applyNormalMatrix(t){return this.applyMatrix3(t).normalize()}applyMatrix4(t){let e=this.x,n=this.y,s=this.z,r=t.elements,a=1/(r[3]*e+r[7]*n+r[11]*s+r[15]);return this.x=(r[0]*e+r[4]*n+r[8]*s+r[12])*a,this.y=(r[1]*e+r[5]*n+r[9]*s+r[13])*a,this.z=(r[2]*e+r[6]*n+r[10]*s+r[14])*a,this}applyQuaternion(t){let e=this.x,n=this.y,s=this.z,r=t.x,a=t.y,o=t.z,c=t.w,l=2*(a*s-o*n),u=2*(o*e-r*s),h=2*(r*n-a*e);return this.x=e+c*l+a*h-o*u,this.y=n+c*u+o*l-r*h,this.z=s+c*h+r*u-a*l,this}project(t){return this.applyMatrix4(t.matrixWorldInverse).applyMatrix4(t.projectionMatrix)}unproject(t){return this.applyMatrix4(t.projectionMatrixInverse).applyMatrix4(t.matrixWorld)}transformDirection(t){let e=this.x,n=this.y,s=this.z,r=t.elements;return this.x=r[0]*e+r[4]*n+r[8]*s,this.y=r[1]*e+r[5]*n+r[9]*s,this.z=r[2]*e+r[6]*n+r[10]*s,this.normalize()}divide(t){return this.x/=t.x,this.y/=t.y,this.z/=t.z,this}divideScalar(t){return this.multiplyScalar(1/t)}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this.z=Math.min(this.z,t.z),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this.z=Math.max(this.z,t.z),this}clamp(t,e){return this.x=ne(this.x,t.x,e.x),this.y=ne(this.y,t.y,e.y),this.z=ne(this.z,t.z,e.z),this}clampScalar(t,e){return this.x=ne(this.x,t,e),this.y=ne(this.y,t,e),this.z=ne(this.z,t,e),this}clampLength(t,e){let n=this.length();return this.divideScalar(n||1).multiplyScalar(ne(n,t,e))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)}normalize(){return this.divideScalar(this.length()||1)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this.z+=(t.z-this.z)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this.z=t.z+(e.z-t.z)*n,this}cross(t){return this.crossVectors(this,t)}crossVectors(t,e){let n=t.x,s=t.y,r=t.z,a=e.x,o=e.y,c=e.z;return this.x=s*c-r*o,this.y=r*a-n*c,this.z=n*o-s*a,this}projectOnVector(t){let e=t.lengthSq();if(e===0)return this.set(0,0,0);let n=t.dot(this)/e;return this.copy(t).multiplyScalar(n)}projectOnPlane(t){return ul.copy(this).projectOnVector(t),this.sub(ul)}reflect(t){return this.sub(ul.copy(t).multiplyScalar(2*this.dot(t)))}angleTo(t){let e=Math.sqrt(this.lengthSq()*t.lengthSq());if(e===0)return Math.PI/2;let n=this.dot(t)/e;return Math.acos(ne(n,-1,1))}distanceTo(t){return Math.sqrt(this.distanceToSquared(t))}distanceToSquared(t){let e=this.x-t.x,n=this.y-t.y,s=this.z-t.z;return e*e+n*n+s*s}manhattanDistanceTo(t){return Math.abs(this.x-t.x)+Math.abs(this.y-t.y)+Math.abs(this.z-t.z)}setFromSpherical(t){return this.setFromSphericalCoords(t.radius,t.phi,t.theta)}setFromSphericalCoords(t,e,n){let s=Math.sin(e)*t;return this.x=s*Math.sin(n),this.y=Math.cos(e)*t,this.z=s*Math.cos(n),this}setFromCylindrical(t){return this.setFromCylindricalCoords(t.radius,t.theta,t.y)}setFromCylindricalCoords(t,e,n){return this.x=t*Math.sin(e),this.y=n,this.z=t*Math.cos(e),this}setFromMatrixPosition(t){let e=t.elements;return this.x=e[12],this.y=e[13],this.z=e[14],this}setFromMatrixScale(t){let e=this.setFromMatrixColumn(t,0).length(),n=this.setFromMatrixColumn(t,1).length(),s=this.setFromMatrixColumn(t,2).length();return this.x=e,this.y=n,this.z=s,this}setFromMatrixColumn(t,e){return this.fromArray(t.elements,e*4)}setFromMatrix3Column(t,e){return this.fromArray(t.elements,e*3)}setFromEuler(t){return this.x=t._x,this.y=t._y,this.z=t._z,this}setFromColor(t){return this.x=t.r,this.y=t.g,this.z=t.b,this}equals(t){return t.x===this.x&&t.y===this.y&&t.z===this.z}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this.z=t[e+2],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t[e+2]=this.z,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this.z=t.getZ(e),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this}randomDirection(){let t=Math.random()*Math.PI*2,e=Math.random()*2-1,n=Math.sqrt(1-e*e);return this.x=n*Math.cos(t),this.y=e,this.z=n*Math.sin(t),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z}},ul=new F,uh=new ln,Xt=class{static{this.prototype.isMatrix3=!0}constructor(t,e,n,s,r,a,o,c,l){this.elements=[1,0,0,0,1,0,0,0,1],t!==void 0&&this.set(t,e,n,s,r,a,o,c,l)}set(t,e,n,s,r,a,o,c,l){let u=this.elements;return u[0]=t,u[1]=s,u[2]=o,u[3]=e,u[4]=r,u[5]=c,u[6]=n,u[7]=a,u[8]=l,this}identity(){return this.set(1,0,0,0,1,0,0,0,1),this}copy(t){let e=this.elements,n=t.elements;return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],this}extractBasis(t,e,n){return t.setFromMatrix3Column(this,0),e.setFromMatrix3Column(this,1),n.setFromMatrix3Column(this,2),this}setFromMatrix4(t){let e=t.elements;return this.set(e[0],e[4],e[8],e[1],e[5],e[9],e[2],e[6],e[10]),this}multiply(t){return this.multiplyMatrices(this,t)}premultiply(t){return this.multiplyMatrices(t,this)}multiplyMatrices(t,e){let n=t.elements,s=e.elements,r=this.elements,a=n[0],o=n[3],c=n[6],l=n[1],u=n[4],h=n[7],d=n[2],f=n[5],m=n[8],y=s[0],g=s[3],p=s[6],v=s[1],b=s[4],_=s[7],T=s[2],w=s[5],x=s[8];return r[0]=a*y+o*v+c*T,r[3]=a*g+o*b+c*w,r[6]=a*p+o*_+c*x,r[1]=l*y+u*v+h*T,r[4]=l*g+u*b+h*w,r[7]=l*p+u*_+h*x,r[2]=d*y+f*v+m*T,r[5]=d*g+f*b+m*w,r[8]=d*p+f*_+m*x,this}multiplyScalar(t){let e=this.elements;return e[0]*=t,e[3]*=t,e[6]*=t,e[1]*=t,e[4]*=t,e[7]*=t,e[2]*=t,e[5]*=t,e[8]*=t,this}determinant(){let t=this.elements,e=t[0],n=t[1],s=t[2],r=t[3],a=t[4],o=t[5],c=t[6],l=t[7],u=t[8];return e*a*u-e*o*l-n*r*u+n*o*c+s*r*l-s*a*c}invert(){let t=this.elements,e=t[0],n=t[1],s=t[2],r=t[3],a=t[4],o=t[5],c=t[6],l=t[7],u=t[8],h=u*a-o*l,d=o*c-u*r,f=l*r-a*c,m=e*h+n*d+s*f;if(m===0)return this.set(0,0,0,0,0,0,0,0,0);let y=1/m;return t[0]=h*y,t[1]=(s*l-u*n)*y,t[2]=(o*n-s*a)*y,t[3]=d*y,t[4]=(u*e-s*c)*y,t[5]=(s*r-o*e)*y,t[6]=f*y,t[7]=(n*c-l*e)*y,t[8]=(a*e-n*r)*y,this}transpose(){let t,e=this.elements;return t=e[1],e[1]=e[3],e[3]=t,t=e[2],e[2]=e[6],e[6]=t,t=e[5],e[5]=e[7],e[7]=t,this}getNormalMatrix(t){return this.setFromMatrix4(t).invert().transpose()}transposeIntoArray(t){let e=this.elements;return t[0]=e[0],t[1]=e[3],t[2]=e[6],t[3]=e[1],t[4]=e[4],t[5]=e[7],t[6]=e[2],t[7]=e[5],t[8]=e[8],this}setUvTransform(t,e,n,s,r,a,o){let c=Math.cos(r),l=Math.sin(r);return this.set(n*c,n*l,-n*(c*a+l*o)+a+t,-s*l,s*c,-s*(-l*a+c*o)+o+e,0,0,1),this}scale(t,e){return Ks("Matrix3: .scale() is deprecated. Use .makeScale() instead."),this.premultiply(dl.makeScale(t,e)),this}rotate(t){return Ks("Matrix3: .rotate() is deprecated. Use .makeRotation() instead."),this.premultiply(dl.makeRotation(-t)),this}translate(t,e){return Ks("Matrix3: .translate() is deprecated. Use .makeTranslation() instead."),this.premultiply(dl.makeTranslation(t,e)),this}makeTranslation(t,e){return t.isVector2?this.set(1,0,t.x,0,1,t.y,0,0,1):this.set(1,0,t,0,1,e,0,0,1),this}makeRotation(t){let e=Math.cos(t),n=Math.sin(t);return this.set(e,-n,0,n,e,0,0,0,1),this}makeScale(t,e){return this.set(t,0,0,0,e,0,0,0,1),this}equals(t){let e=this.elements,n=t.elements;for(let s=0;s<9;s++)if(e[s]!==n[s])return!1;return!0}fromArray(t,e=0){for(let n=0;n<9;n++)this.elements[n]=t[n+e];return this}toArray(t=[],e=0){let n=this.elements;return t[e]=n[0],t[e+1]=n[1],t[e+2]=n[2],t[e+3]=n[3],t[e+4]=n[4],t[e+5]=n[5],t[e+6]=n[6],t[e+7]=n[7],t[e+8]=n[8],t}clone(){return new this.constructor().fromArray(this.elements)}},dl=new Xt,dh=new Xt().set(.4123908,.3575843,.1804808,.212639,.7151687,.0721923,.0193308,.1191948,.9505322),fh=new Xt().set(3.2409699,-1.5373832,-.4986108,-.9692436,1.8759675,.0415551,.0556301,-.203977,1.0569715);function Sf(){let i={enabled:!0,workingColorSpace:ir,spaces:{},convert:function(s,r,a){return this.enabled===!1||r===a||!r||!a||(this.spaces[r].transfer===ge&&(s.r=Jn(s.r),s.g=Jn(s.g),s.b=Jn(s.b)),this.spaces[r].primaries!==this.spaces[a].primaries&&(s.applyMatrix3(this.spaces[r].toXYZ),s.applyMatrix3(this.spaces[a].fromXYZ)),this.spaces[a].transfer===ge&&(s.r=fs(s.r),s.g=fs(s.g),s.b=fs(s.b))),s},workingToColorSpace:function(s,r){return this.convert(s,this.workingColorSpace,r)},colorSpaceToWorking:function(s,r){return this.convert(s,r,this.workingColorSpace)},getPrimaries:function(s){return this.spaces[s].primaries},getTransfer:function(s){return s===ai?sr:this.spaces[s].transfer},getToneMappingMode:function(s){return this.spaces[s].outputColorSpaceConfig.toneMappingMode||"standard"},getLuminanceCoefficients:function(s,r=this.workingColorSpace){return s.fromArray(this.spaces[r].luminanceCoefficients)},define:function(s){Object.assign(this.spaces,s)},_getMatrix:function(s,r,a){return s.copy(this.spaces[a].fromXYZ).multiply(this.spaces[r].toXYZ)},_getDrawingBufferColorSpace:function(s){return this.spaces[s].outputColorSpaceConfig.drawingBufferColorSpace},_getUnpackColorSpace:function(s=this.workingColorSpace){return this.spaces[s].workingColorSpaceConfig.unpackColorSpace}},t=[.64,.33,.3,.6,.15,.06],e=[.2126,.7152,.0722],n=[.3127,.329];return i.define({[ir]:{primaries:t,whitePoint:n,transfer:sr,toXYZ:dh,fromXYZ:fh,luminanceCoefficients:e,workingColorSpaceConfig:{unpackColorSpace:He},outputColorSpaceConfig:{drawingBufferColorSpace:He}},[He]:{primaries:t,whitePoint:n,transfer:ge,toXYZ:dh,fromXYZ:fh,luminanceCoefficients:e,outputColorSpaceConfig:{drawingBufferColorSpace:He}}}),i}var oe=Sf();function Jn(i){return i<.04045?i*.0773993808:Math.pow(i*.9478672986+.0521327014,2.4)}function fs(i){return i<.0031308?i*12.92:1.055*Math.pow(i,.41666)-.055}var Qi,Oa=class{static getDataURL(t,e="image/png"){if(/^data:/i.test(t.src)||typeof HTMLCanvasElement>"u")return t.src;let n;if(t instanceof HTMLCanvasElement)n=t;else{Qi===void 0&&(Qi=rr("canvas")),Qi.width=t.width,Qi.height=t.height;let s=Qi.getContext("2d");t instanceof ImageData?s.putImageData(t,0,0):s.drawImage(t,0,0,t.width,t.height),n=Qi}return n.toDataURL(e)}static sRGBToLinear(t){if(typeof HTMLImageElement<"u"&&t instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&t instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&t instanceof ImageBitmap){let e=rr("canvas");e.width=t.width,e.height=t.height;let n=e.getContext("2d");n.drawImage(t,0,0,t.width,t.height);let s=n.getImageData(0,0,t.width,t.height),r=s.data;for(let a=0;a<r.length;a++)r[a]=Jn(r[a]/255)*255;return n.putImageData(s,0,0),e}else if(t.data){let e=t.data.slice(0);for(let n=0;n<e.length;n++)e instanceof Uint8Array||e instanceof Uint8ClampedArray?e[n]=Math.floor(Jn(e[n]/255)*255):e[n]=Jn(e[n]);return{data:e,width:t.width,height:t.height}}else return qt("ImageUtils.sRGBToLinear(): Unsupported image type. No color space conversion applied."),t}},wf=0,vs=class{constructor(t=null){this.isTextureSource=!0,Object.defineProperty(this,"id",{value:wf++}),this.uuid=Zi(),this.data=t,this.dataReady=!0,this.version=0}getSize(t){let e=this.data;return typeof HTMLVideoElement<"u"&&e instanceof HTMLVideoElement?t.set(e.videoWidth,e.videoHeight,0):typeof VideoFrame<"u"&&e instanceof VideoFrame?t.set(e.displayWidth,e.displayHeight,0):e!==null?t.set(e.width||0,e.height||0,e.depth||0):t.set(0,0,0),t}set needsUpdate(t){t===!0&&this.version++}toJSON(t){let e=t===void 0||typeof t=="string";if(!e&&t.images[this.uuid]!==void 0)return t.images[this.uuid];let n={uuid:this.uuid,url:""},s=this.data;if(s!==null){let r;if(Array.isArray(s)){r=[];for(let a=0,o=s.length;a<o;a++)s[a].isDataTexture?r.push(fl(s[a].image)):r.push(fl(s[a]))}else r=fl(s);n.url=r}return e||(t.images[this.uuid]=n),n}};function fl(i){return typeof HTMLImageElement<"u"&&i instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&i instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&i instanceof ImageBitmap?Oa.getDataURL(i):i.data?{data:Array.from(i.data),width:i.width,height:i.height,type:i.data.constructor.name}:(qt("Texture: Unable to serialize Texture."),{})}var Tf=0,pl=new F,Qe=class i extends Sn{constructor(t=i.DEFAULT_IMAGE,e=i.DEFAULT_MAPPING,n=mn,s=mn,r=je,a=si,o=Dn,c=nn,l=i.DEFAULT_ANISOTROPY,u=ai){super(),this.isTexture=!0,Object.defineProperty(this,"id",{value:Tf++}),this.uuid=Zi(),this.name="",this.source=new vs(t),this.mipmaps=[],this.mapping=e,this.channel=0,this.wrapS=n,this.wrapT=s,this.magFilter=r,this.minFilter=a,this.anisotropy=l,this.format=o,this.internalFormat=null,this.type=c,this.offset=new ft(0,0),this.repeat=new ft(1,1),this.center=new ft(0,0),this.rotation=0,this.matrixAutoUpdate=!0,this.matrix=new Xt,this.generateMipmaps=!0,this.mipmapsAutoUpdate=!0,this.premultiplyAlpha=!1,this.flipY=!0,this.unpackAlignment=4,this.colorSpace=u,this.userData={},this.updateRanges=[],this.version=0,this.onUpdate=null,this.renderTarget=null,this.isRenderTargetTexture=!1,this.isArrayTexture=!!(t&&t.depth&&t.depth>1),this.pmremVersion=0,this.isPMREMTexture=!1,this.normalized=!1}get width(){return this.source.getSize(pl).x}get height(){return this.source.getSize(pl).y}get depth(){return this.source.getSize(pl).z}get image(){return this.source.data}set image(t){this.source.data=t}updateMatrix(){this.matrix.setUvTransform(this.offset.x,this.offset.y,this.repeat.x,this.repeat.y,this.rotation,this.center.x,this.center.y)}addUpdateRange(t,e){this.updateRanges.push({start:t,count:e})}clearUpdateRanges(){this.updateRanges.length=0}clone(){return new this.constructor().copy(this)}copy(t){return this.name=t.name,this.source=t.source,this.mipmaps=t.mipmaps.slice(0),this.mapping=t.mapping,this.channel=t.channel,this.wrapS=t.wrapS,this.wrapT=t.wrapT,this.magFilter=t.magFilter,this.minFilter=t.minFilter,this.anisotropy=t.anisotropy,this.format=t.format,this.internalFormat=t.internalFormat,this.type=t.type,this.normalized=t.normalized,this.offset.copy(t.offset),this.repeat.copy(t.repeat),this.center.copy(t.center),this.rotation=t.rotation,this.matrixAutoUpdate=t.matrixAutoUpdate,this.matrix.copy(t.matrix),this.generateMipmaps=t.generateMipmaps,this.mipmapsAutoUpdate=t.mipmapsAutoUpdate,this.premultiplyAlpha=t.premultiplyAlpha,this.flipY=t.flipY,this.unpackAlignment=t.unpackAlignment,this.colorSpace=t.colorSpace,this.renderTarget=t.renderTarget,this.isRenderTargetTexture=t.isRenderTargetTexture,this.isPMREMTexture=t.isPMREMTexture,this.isArrayTexture=t.isArrayTexture,this.userData=JSON.parse(JSON.stringify(t.userData)),this.needsUpdate=!0,this}setValues(t){for(let e in t){let n=t[e];if(n===void 0){qt(`Texture.setValues(): parameter '${e}' has value of undefined.`);continue}let s=this[e];if(s===void 0){qt(`Texture.setValues(): property '${e}' does not exist.`);continue}s&&n&&s.isVector2&&n.isVector2||s&&n&&s.isVector3&&n.isVector3||s&&n&&s.isMatrix3&&n.isMatrix3?s.copy(n):this[e]=n}}toJSON(t){let e=t===void 0||typeof t=="string";if(!e&&t.textures[this.uuid]!==void 0)return t.textures[this.uuid];let n={metadata:{version:4.7,type:"Texture",generator:"Texture.toJSON"},uuid:this.uuid,name:this.name,image:this.source.toJSON(t).uuid,mapping:this.mapping,channel:this.channel,repeat:[this.repeat.x,this.repeat.y],offset:[this.offset.x,this.offset.y],center:[this.center.x,this.center.y],rotation:this.rotation,wrap:[this.wrapS,this.wrapT],format:this.format,internalFormat:this.internalFormat,type:this.type,normalized:this.normalized,colorSpace:this.colorSpace,minFilter:this.minFilter,magFilter:this.magFilter,anisotropy:this.anisotropy,flipY:this.flipY,generateMipmaps:this.generateMipmaps,mipmapsAutoUpdate:this.mipmapsAutoUpdate,premultiplyAlpha:this.premultiplyAlpha,unpackAlignment:this.unpackAlignment};return Object.keys(this.userData).length>0&&(n.userData=this.userData),e||(t.textures[this.uuid]=n),n}dispose(){this.dispatchEvent({type:"dispose"})}transformUv(t){if(this.mapping!==lc)return t;if(t.applyMatrix3(this.matrix),t.x<0||t.x>1)switch(this.wrapS){case ms:t.x=t.x-Math.floor(t.x);break;case mn:t.x=t.x<0?0:1;break;case gs:Math.abs(Math.floor(t.x)%2)===1?t.x=Math.ceil(t.x)-t.x:t.x=t.x-Math.floor(t.x);break}if(t.y<0||t.y>1)switch(this.wrapT){case ms:t.y=t.y-Math.floor(t.y);break;case mn:t.y=t.y<0?0:1;break;case gs:Math.abs(Math.floor(t.y)%2)===1?t.y=Math.ceil(t.y)-t.y:t.y=t.y-Math.floor(t.y);break}return this.flipY&&(t.y=1-t.y),t}set needsUpdate(t){t===!0&&(this.version++,this.source.needsUpdate=!0)}set needsPMREMUpdate(t){t===!0&&this.pmremVersion++}};Qe.DEFAULT_IMAGE=null;Qe.DEFAULT_MAPPING=lc;Qe.DEFAULT_ANISOTROPY=1;var Se=class{static{this.prototype.isVector4=!0}constructor(t=0,e=0,n=0,s=1){this.x=t,this.y=e,this.z=n,this.w=s}get width(){return this.z}set width(t){this.z=t}get height(){return this.w}set height(t){this.w=t}set(t,e,n,s){return this.x=t,this.y=e,this.z=n,this.w=s,this}setScalar(t){return this.x=t,this.y=t,this.z=t,this.w=t,this}setX(t){return this.x=t,this}setY(t){return this.y=t,this}setZ(t){return this.z=t,this}setW(t){return this.w=t,this}setComponent(t,e){switch(t){case 0:this.x=e;break;case 1:this.y=e;break;case 2:this.z=e;break;case 3:this.w=e;break;default:throw new Error("THREE.Vector4: index is out of range: "+t)}return this}getComponent(t){switch(t){case 0:return this.x;case 1:return this.y;case 2:return this.z;case 3:return this.w;default:throw new Error("THREE.Vector4: index is out of range: "+t)}}clone(){return new this.constructor(this.x,this.y,this.z,this.w)}copy(t){return this.x=t.x,this.y=t.y,this.z=t.z,this.w=t.w!==void 0?t.w:1,this}add(t){return this.x+=t.x,this.y+=t.y,this.z+=t.z,this.w+=t.w,this}addScalar(t){return this.x+=t,this.y+=t,this.z+=t,this.w+=t,this}addVectors(t,e){return this.x=t.x+e.x,this.y=t.y+e.y,this.z=t.z+e.z,this.w=t.w+e.w,this}addScaledVector(t,e){return this.x+=t.x*e,this.y+=t.y*e,this.z+=t.z*e,this.w+=t.w*e,this}sub(t){return this.x-=t.x,this.y-=t.y,this.z-=t.z,this.w-=t.w,this}subScalar(t){return this.x-=t,this.y-=t,this.z-=t,this.w-=t,this}subVectors(t,e){return this.x=t.x-e.x,this.y=t.y-e.y,this.z=t.z-e.z,this.w=t.w-e.w,this}multiply(t){return this.x*=t.x,this.y*=t.y,this.z*=t.z,this.w*=t.w,this}multiplyScalar(t){return this.x*=t,this.y*=t,this.z*=t,this.w*=t,this}applyMatrix4(t){let e=this.x,n=this.y,s=this.z,r=this.w,a=t.elements;return this.x=a[0]*e+a[4]*n+a[8]*s+a[12]*r,this.y=a[1]*e+a[5]*n+a[9]*s+a[13]*r,this.z=a[2]*e+a[6]*n+a[10]*s+a[14]*r,this.w=a[3]*e+a[7]*n+a[11]*s+a[15]*r,this}divide(t){return this.x/=t.x,this.y/=t.y,this.z/=t.z,this.w/=t.w,this}divideScalar(t){return this.multiplyScalar(1/t)}setAxisAngleFromQuaternion(t){this.w=2*Math.acos(t.w);let e=Math.sqrt(1-t.w*t.w);return e<1e-4?(this.x=1,this.y=0,this.z=0):(this.x=t.x/e,this.y=t.y/e,this.z=t.z/e),this}setAxisAngleFromRotationMatrix(t){let e,n,s,r,c=t.elements,l=c[0],u=c[4],h=c[8],d=c[1],f=c[5],m=c[9],y=c[2],g=c[6],p=c[10];if(Math.abs(u-d)<.01&&Math.abs(h-y)<.01&&Math.abs(m-g)<.01){if(Math.abs(u+d)<.1&&Math.abs(h+y)<.1&&Math.abs(m+g)<.1&&Math.abs(l+f+p-3)<.1)return this.set(1,0,0,0),this;e=Math.PI;let b=(l+1)/2,_=(f+1)/2,T=(p+1)/2,w=(u+d)/4,x=(h+y)/4,M=(m+g)/4;return b>_&&b>T?b<.01?(n=0,s=.707106781,r=.707106781):(n=Math.sqrt(b),s=w/n,r=x/n):_>T?_<.01?(n=.707106781,s=0,r=.707106781):(s=Math.sqrt(_),n=w/s,r=M/s):T<.01?(n=.707106781,s=.707106781,r=0):(r=Math.sqrt(T),n=x/r,s=M/r),this.set(n,s,r,e),this}let v=Math.sqrt((g-m)*(g-m)+(h-y)*(h-y)+(d-u)*(d-u));return Math.abs(v)<.001&&(v=1),this.x=(g-m)/v,this.y=(h-y)/v,this.z=(d-u)/v,this.w=Math.acos((l+f+p-1)/2),this}setFromMatrixPosition(t){let e=t.elements;return this.x=e[12],this.y=e[13],this.z=e[14],this.w=e[15],this}min(t){return this.x=Math.min(this.x,t.x),this.y=Math.min(this.y,t.y),this.z=Math.min(this.z,t.z),this.w=Math.min(this.w,t.w),this}max(t){return this.x=Math.max(this.x,t.x),this.y=Math.max(this.y,t.y),this.z=Math.max(this.z,t.z),this.w=Math.max(this.w,t.w),this}clamp(t,e){return this.x=ne(this.x,t.x,e.x),this.y=ne(this.y,t.y,e.y),this.z=ne(this.z,t.z,e.z),this.w=ne(this.w,t.w,e.w),this}clampScalar(t,e){return this.x=ne(this.x,t,e),this.y=ne(this.y,t,e),this.z=ne(this.z,t,e),this.w=ne(this.w,t,e),this}clampLength(t,e){let n=this.length();return this.divideScalar(n||1).multiplyScalar(ne(n,t,e))}floor(){return this.x=Math.floor(this.x),this.y=Math.floor(this.y),this.z=Math.floor(this.z),this.w=Math.floor(this.w),this}ceil(){return this.x=Math.ceil(this.x),this.y=Math.ceil(this.y),this.z=Math.ceil(this.z),this.w=Math.ceil(this.w),this}round(){return this.x=Math.round(this.x),this.y=Math.round(this.y),this.z=Math.round(this.z),this.w=Math.round(this.w),this}roundToZero(){return this.x=Math.trunc(this.x),this.y=Math.trunc(this.y),this.z=Math.trunc(this.z),this.w=Math.trunc(this.w),this}negate(){return this.x=-this.x,this.y=-this.y,this.z=-this.z,this.w=-this.w,this}dot(t){return this.x*t.x+this.y*t.y+this.z*t.z+this.w*t.w}lengthSq(){return this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w}length(){return Math.sqrt(this.x*this.x+this.y*this.y+this.z*this.z+this.w*this.w)}manhattanLength(){return Math.abs(this.x)+Math.abs(this.y)+Math.abs(this.z)+Math.abs(this.w)}normalize(){return this.divideScalar(this.length()||1)}setLength(t){return this.normalize().multiplyScalar(t)}lerp(t,e){return this.x+=(t.x-this.x)*e,this.y+=(t.y-this.y)*e,this.z+=(t.z-this.z)*e,this.w+=(t.w-this.w)*e,this}lerpVectors(t,e,n){return this.x=t.x+(e.x-t.x)*n,this.y=t.y+(e.y-t.y)*n,this.z=t.z+(e.z-t.z)*n,this.w=t.w+(e.w-t.w)*n,this}equals(t){return t.x===this.x&&t.y===this.y&&t.z===this.z&&t.w===this.w}fromArray(t,e=0){return this.x=t[e],this.y=t[e+1],this.z=t[e+2],this.w=t[e+3],this}toArray(t=[],e=0){return t[e]=this.x,t[e+1]=this.y,t[e+2]=this.z,t[e+3]=this.w,t}fromBufferAttribute(t,e){return this.x=t.getX(e),this.y=t.getY(e),this.z=t.getZ(e),this.w=t.getW(e),this}random(){return this.x=Math.random(),this.y=Math.random(),this.z=Math.random(),this.w=Math.random(),this}*[Symbol.iterator](){yield this.x,yield this.y,yield this.z,yield this.w}},Ba=class extends Sn{constructor(t=1,e=1,n={}){super(),n=Object.assign({generateMipmaps:!1,internalFormat:null,minFilter:je,depthBuffer:!0,stencilBuffer:!1,resolveColorBuffer:!0,resolveDepthBuffer:!0,resolveStencilBuffer:!0,storeMultisampledColorBuffer:!0,storeMultisampledDepthBuffer:!0,storeMultisampledStencilBuffer:!0,depthTexture:null,samples:0,count:1,depth:1,multiview:!1,useArrayDepthTexture:!1},n),this.isRenderTarget=!0,this.width=t,this.height=e,this.depth=n.depth,this.scissor=new Se(0,0,t,e),this.scissorTest=!1,this.viewport=new Se(0,0,t,e),this.textures=[];let s=n.count;for(let r=0;r<s;r++)this.textures[r]=new Qe({width:t,height:e,depth:n.depth}),this.textures[r].isRenderTargetTexture=!0,this.textures[r].renderTarget=this;this._setTextureOptions(n),this.depthBuffer=n.depthBuffer,this.stencilBuffer=n.stencilBuffer,this.resolveColorBuffer=n.resolveColorBuffer,this.resolveDepthBuffer=n.resolveDepthBuffer,this.resolveStencilBuffer=n.resolveStencilBuffer,this.storeMultisampledColorBuffer=n.storeMultisampledColorBuffer,this.storeMultisampledDepthBuffer=n.storeMultisampledDepthBuffer,this.storeMultisampledStencilBuffer=n.storeMultisampledStencilBuffer,this._depthTexture=null,this.depthTexture=n.depthTexture,this.samples=n.samples,this.multiview=n.multiview,this.useArrayDepthTexture=n.useArrayDepthTexture}_setTextureOptions(t={}){let e={minFilter:je,generateMipmaps:!1,flipY:!1,internalFormat:null};t.mapping!==void 0&&(e.mapping=t.mapping),t.wrapS!==void 0&&(e.wrapS=t.wrapS),t.wrapT!==void 0&&(e.wrapT=t.wrapT),t.wrapR!==void 0&&(e.wrapR=t.wrapR),t.magFilter!==void 0&&(e.magFilter=t.magFilter),t.minFilter!==void 0&&(e.minFilter=t.minFilter),t.format!==void 0&&(e.format=t.format),t.type!==void 0&&(e.type=t.type),t.anisotropy!==void 0&&(e.anisotropy=t.anisotropy),t.colorSpace!==void 0&&(e.colorSpace=t.colorSpace),t.flipY!==void 0&&(e.flipY=t.flipY),t.generateMipmaps!==void 0&&(e.generateMipmaps=t.generateMipmaps),t.mipmapsAutoUpdate!==void 0&&(e.mipmapsAutoUpdate=t.mipmapsAutoUpdate),t.internalFormat!==void 0&&(e.internalFormat=t.internalFormat);for(let n=0;n<this.textures.length;n++)this.textures[n].setValues(e)}get texture(){return this.textures[0]}set texture(t){this.textures[0]=t}set depthTexture(t){this._depthTexture!==null&&this._depthTexture.renderTarget===this&&(this._depthTexture.renderTarget=null),t!==null&&t.renderTarget===null&&(t.renderTarget=this),this._depthTexture=t}get depthTexture(){return this._depthTexture}setSize(t,e,n=1){if(this.width!==t||this.height!==e||this.depth!==n){this.width=t,this.height=e,this.depth=n;for(let s=0,r=this.textures.length;s<r;s++)this.textures[s].image.width=t,this.textures[s].image.height=e,this.textures[s].image.depth=n,this.textures[s].isData3DTexture!==!0&&(this.textures[s].isArrayTexture=this.textures[s].image.depth>1);this.dispose()}this.viewport.set(0,0,t,e),this.scissor.set(0,0,t,e)}clone(){return new this.constructor().copy(this)}copy(t){this.width=t.width,this.height=t.height,this.depth=t.depth,this.scissor.copy(t.scissor),this.scissorTest=t.scissorTest,this.viewport.copy(t.viewport),this.textures.length=0;for(let e=0,n=t.textures.length;e<n;e++){this.textures[e]=t.textures[e].clone(),this.textures[e].isRenderTargetTexture=!0,this.textures[e].renderTarget=this;let s=Object.assign({},t.textures[e].image);this.textures[e].source=new vs(s)}if(this.depthBuffer=t.depthBuffer,this.stencilBuffer=t.stencilBuffer,this.resolveColorBuffer=t.resolveColorBuffer,this.resolveDepthBuffer=t.resolveDepthBuffer,this.resolveStencilBuffer=t.resolveStencilBuffer,this.storeMultisampledColorBuffer=t.storeMultisampledColorBuffer,this.storeMultisampledDepthBuffer=t.storeMultisampledDepthBuffer,this.storeMultisampledStencilBuffer=t.storeMultisampledStencilBuffer,t.depthTexture!==null)if(t.depthTexture.renderTarget===t){let e=t.depthTexture.clone();e.renderTarget=null,this.depthTexture=e}else this.depthTexture=t.depthTexture;else this.depthTexture=null;return this.samples=t.samples,this.multiview=t.multiview,this.useArrayDepthTexture=t.useArrayDepthTexture,this}dispose(){this.dispatchEvent({type:"dispose"})}},cn=class extends Ba{constructor(t=1,e=1,n={}){super(t,e,n),this.isWebGLRenderTarget=!0}},ar=class extends Qe{constructor(t=null,e=1,n=1,s=1){super(null),this.isDataArrayTexture=!0,this.image={data:t,width:e,height:n,depth:s},this.magFilter=Ge,this.minFilter=Ge,this.wrapR=mn,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1,this.layerUpdates=new Set}copy(t){return super.copy(t),this.wrapR=t.wrapR,this}addLayerUpdate(t){this.layerUpdates.add(t)}clearLayerUpdates(){this.layerUpdates.clear()}};var ka=class extends Qe{constructor(t=null,e=1,n=1,s=1){super(null),this.isData3DTexture=!0,this.image={data:t,width:e,height:n,depth:s},this.magFilter=Ge,this.minFilter=Ge,this.wrapR=mn,this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}copy(t){return super.copy(t),this.wrapR=t.wrapR,this}};var xe=class i{static{this.prototype.isMatrix4=!0}constructor(t,e,n,s,r,a,o,c,l,u,h,d,f,m,y,g){this.elements=[1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1],t!==void 0&&this.set(t,e,n,s,r,a,o,c,l,u,h,d,f,m,y,g)}set(t,e,n,s,r,a,o,c,l,u,h,d,f,m,y,g){let p=this.elements;return p[0]=t,p[4]=e,p[8]=n,p[12]=s,p[1]=r,p[5]=a,p[9]=o,p[13]=c,p[2]=l,p[6]=u,p[10]=h,p[14]=d,p[3]=f,p[7]=m,p[11]=y,p[15]=g,this}identity(){return this.set(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1),this}clone(){return new i().fromArray(this.elements)}copy(t){let e=this.elements,n=t.elements;return e[0]=n[0],e[1]=n[1],e[2]=n[2],e[3]=n[3],e[4]=n[4],e[5]=n[5],e[6]=n[6],e[7]=n[7],e[8]=n[8],e[9]=n[9],e[10]=n[10],e[11]=n[11],e[12]=n[12],e[13]=n[13],e[14]=n[14],e[15]=n[15],this}copyPosition(t){let e=this.elements,n=t.elements;return e[12]=n[12],e[13]=n[13],e[14]=n[14],this}setFromMatrix3(t){let e=t.elements;return this.set(e[0],e[3],e[6],0,e[1],e[4],e[7],0,e[2],e[5],e[8],0,0,0,0,1),this}extractBasis(t,e,n){return this.determinantAffine()===0?(t.set(1,0,0),e.set(0,1,0),n.set(0,0,1),this):(t.setFromMatrixColumn(this,0),e.setFromMatrixColumn(this,1),n.setFromMatrixColumn(this,2),this)}makeBasis(t,e,n){return this.set(t.x,e.x,n.x,0,t.y,e.y,n.y,0,t.z,e.z,n.z,0,0,0,0,1),this}extractRotation(t){if(t.determinantAffine()===0)return this.identity();let e=this.elements,n=t.elements,s=1/ts.setFromMatrixColumn(t,0).length(),r=1/ts.setFromMatrixColumn(t,1).length(),a=1/ts.setFromMatrixColumn(t,2).length();return e[0]=n[0]*s,e[1]=n[1]*s,e[2]=n[2]*s,e[3]=0,e[4]=n[4]*r,e[5]=n[5]*r,e[6]=n[6]*r,e[7]=0,e[8]=n[8]*a,e[9]=n[9]*a,e[10]=n[10]*a,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,this}makeRotationFromEuler(t){let e=this.elements,n=t.x,s=t.y,r=t.z,a=Math.cos(n),o=Math.sin(n),c=Math.cos(s),l=Math.sin(s),u=Math.cos(r),h=Math.sin(r);if(t.order==="XYZ"){let d=a*u,f=a*h,m=o*u,y=o*h;e[0]=c*u,e[4]=-c*h,e[8]=l,e[1]=f+m*l,e[5]=d-y*l,e[9]=-o*c,e[2]=y-d*l,e[6]=m+f*l,e[10]=a*c}else if(t.order==="YXZ"){let d=c*u,f=c*h,m=l*u,y=l*h;e[0]=d+y*o,e[4]=m*o-f,e[8]=a*l,e[1]=a*h,e[5]=a*u,e[9]=-o,e[2]=f*o-m,e[6]=y+d*o,e[10]=a*c}else if(t.order==="ZXY"){let d=c*u,f=c*h,m=l*u,y=l*h;e[0]=d-y*o,e[4]=-a*h,e[8]=m+f*o,e[1]=f+m*o,e[5]=a*u,e[9]=y-d*o,e[2]=-a*l,e[6]=o,e[10]=a*c}else if(t.order==="ZYX"){let d=a*u,f=a*h,m=o*u,y=o*h;e[0]=c*u,e[4]=m*l-f,e[8]=d*l+y,e[1]=c*h,e[5]=y*l+d,e[9]=f*l-m,e[2]=-l,e[6]=o*c,e[10]=a*c}else if(t.order==="YZX"){let d=a*c,f=a*l,m=o*c,y=o*l;e[0]=c*u,e[4]=y-d*h,e[8]=m*h+f,e[1]=h,e[5]=a*u,e[9]=-o*u,e[2]=-l*u,e[6]=f*h+m,e[10]=d-y*h}else if(t.order==="XZY"){let d=a*c,f=a*l,m=o*c,y=o*l;e[0]=c*u,e[4]=-h,e[8]=l*u,e[1]=d*h+y,e[5]=a*u,e[9]=f*h-m,e[2]=m*h-f,e[6]=o*u,e[10]=y*h+d}return e[3]=0,e[7]=0,e[11]=0,e[12]=0,e[13]=0,e[14]=0,e[15]=1,this}makeRotationFromQuaternion(t){return this.compose(Af,t,Ef)}lookAt(t,e,n){let s=this.elements;return rn.subVectors(t,e),rn.lengthSq()===0&&(rn.z=1),rn.normalize(),ui.crossVectors(n,rn),ui.lengthSq()===0&&(Math.abs(n.z)===1?rn.x+=1e-4:rn.z+=1e-4,rn.normalize(),ui.crossVectors(n,rn)),ui.normalize(),na.crossVectors(rn,ui),s[0]=ui.x,s[4]=na.x,s[8]=rn.x,s[1]=ui.y,s[5]=na.y,s[9]=rn.y,s[2]=ui.z,s[6]=na.z,s[10]=rn.z,this}multiply(t){return this.multiplyMatrices(this,t)}premultiply(t){return this.multiplyMatrices(t,this)}multiplyMatrices(t,e){let n=t.elements,s=e.elements,r=this.elements,a=n[0],o=n[4],c=n[8],l=n[12],u=n[1],h=n[5],d=n[9],f=n[13],m=n[2],y=n[6],g=n[10],p=n[14],v=n[3],b=n[7],_=n[11],T=n[15],w=s[0],x=s[4],M=s[8],P=s[12],C=s[1],k=s[5],O=s[9],j=s[13],D=s[2],K=s[6],Q=s[10],Y=s[14],J=s[3],at=s[7],W=s[11],tt=s[15];return r[0]=a*w+o*C+c*D+l*J,r[4]=a*x+o*k+c*K+l*at,r[8]=a*M+o*O+c*Q+l*W,r[12]=a*P+o*j+c*Y+l*tt,r[1]=u*w+h*C+d*D+f*J,r[5]=u*x+h*k+d*K+f*at,r[9]=u*M+h*O+d*Q+f*W,r[13]=u*P+h*j+d*Y+f*tt,r[2]=m*w+y*C+g*D+p*J,r[6]=m*x+y*k+g*K+p*at,r[10]=m*M+y*O+g*Q+p*W,r[14]=m*P+y*j+g*Y+p*tt,r[3]=v*w+b*C+_*D+T*J,r[7]=v*x+b*k+_*K+T*at,r[11]=v*M+b*O+_*Q+T*W,r[15]=v*P+b*j+_*Y+T*tt,this}multiplyScalar(t){let e=this.elements;return e[0]*=t,e[4]*=t,e[8]*=t,e[12]*=t,e[1]*=t,e[5]*=t,e[9]*=t,e[13]*=t,e[2]*=t,e[6]*=t,e[10]*=t,e[14]*=t,e[3]*=t,e[7]*=t,e[11]*=t,e[15]*=t,this}determinant(){let t=this.elements,e=t[0],n=t[4],s=t[8],r=t[12],a=t[1],o=t[5],c=t[9],l=t[13],u=t[2],h=t[6],d=t[10],f=t[14],m=t[3],y=t[7],g=t[11],p=t[15],v=c*f-l*d,b=o*f-l*h,_=o*d-c*h,T=a*f-l*u,w=a*d-c*u,x=a*h-o*u;return e*(y*v-g*b+p*_)-n*(m*v-g*T+p*w)+s*(m*b-y*T+p*x)-r*(m*_-y*w+g*x)}determinantAffine(){let t=this.elements,e=t[0],n=t[4],s=t[8],r=t[1],a=t[5],o=t[9],c=t[2],l=t[6],u=t[10];return e*(a*u-o*l)-n*(r*u-o*c)+s*(r*l-a*c)}transpose(){let t=this.elements,e;return e=t[1],t[1]=t[4],t[4]=e,e=t[2],t[2]=t[8],t[8]=e,e=t[6],t[6]=t[9],t[9]=e,e=t[3],t[3]=t[12],t[12]=e,e=t[7],t[7]=t[13],t[13]=e,e=t[11],t[11]=t[14],t[14]=e,this}setPosition(t,e,n){let s=this.elements;return t.isVector3?(s[12]=t.x,s[13]=t.y,s[14]=t.z):(s[12]=t,s[13]=e,s[14]=n),this}invert(){let t=this.elements,e=t[0],n=t[1],s=t[2],r=t[3],a=t[4],o=t[5],c=t[6],l=t[7],u=t[8],h=t[9],d=t[10],f=t[11],m=t[12],y=t[13],g=t[14],p=t[15],v=e*o-n*a,b=e*c-s*a,_=e*l-r*a,T=n*c-s*o,w=n*l-r*o,x=s*l-r*c,M=u*y-h*m,P=u*g-d*m,C=u*p-f*m,k=h*g-d*y,O=h*p-f*y,j=d*p-f*g,D=v*j-b*O+_*k+T*C-w*P+x*M;if(D===0)return this.set(0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0);let K=1/D;return t[0]=(o*j-c*O+l*k)*K,t[1]=(s*O-n*j-r*k)*K,t[2]=(y*x-g*w+p*T)*K,t[3]=(d*w-h*x-f*T)*K,t[4]=(c*C-a*j-l*P)*K,t[5]=(e*j-s*C+r*P)*K,t[6]=(g*_-m*x-p*b)*K,t[7]=(u*x-d*_+f*b)*K,t[8]=(a*O-o*C+l*M)*K,t[9]=(n*C-e*O-r*M)*K,t[10]=(m*w-y*_+p*v)*K,t[11]=(h*_-u*w-f*v)*K,t[12]=(o*P-a*k-c*M)*K,t[13]=(e*k-n*P+s*M)*K,t[14]=(y*b-m*T-g*v)*K,t[15]=(u*T-h*b+d*v)*K,this}scale(t){let e=this.elements,n=t.x,s=t.y,r=t.z;return e[0]*=n,e[4]*=s,e[8]*=r,e[1]*=n,e[5]*=s,e[9]*=r,e[2]*=n,e[6]*=s,e[10]*=r,e[3]*=n,e[7]*=s,e[11]*=r,this}getMaxScaleOnAxis(){let t=this.elements,e=t[0]*t[0]+t[1]*t[1]+t[2]*t[2],n=t[4]*t[4]+t[5]*t[5]+t[6]*t[6],s=t[8]*t[8]+t[9]*t[9]+t[10]*t[10];return Math.sqrt(Math.max(e,n,s))}makeTranslation(t,e,n){return t.isVector3?this.set(1,0,0,t.x,0,1,0,t.y,0,0,1,t.z,0,0,0,1):this.set(1,0,0,t,0,1,0,e,0,0,1,n,0,0,0,1),this}makeRotationX(t){let e=Math.cos(t),n=Math.sin(t);return this.set(1,0,0,0,0,e,-n,0,0,n,e,0,0,0,0,1),this}makeRotationY(t){let e=Math.cos(t),n=Math.sin(t);return this.set(e,0,n,0,0,1,0,0,-n,0,e,0,0,0,0,1),this}makeRotationZ(t){let e=Math.cos(t),n=Math.sin(t);return this.set(e,-n,0,0,n,e,0,0,0,0,1,0,0,0,0,1),this}makeRotationAxis(t,e){let n=Math.cos(e),s=Math.sin(e),r=1-n,a=t.x,o=t.y,c=t.z,l=r*a,u=r*o;return this.set(l*a+n,l*o-s*c,l*c+s*o,0,l*o+s*c,u*o+n,u*c-s*a,0,l*c-s*o,u*c+s*a,r*c*c+n,0,0,0,0,1),this}makeScale(t,e,n){return this.set(t,0,0,0,0,e,0,0,0,0,n,0,0,0,0,1),this}makeShear(t,e,n,s,r,a){return this.set(1,n,r,0,t,1,a,0,e,s,1,0,0,0,0,1),this}compose(t,e,n){let s=this.elements,r=e._x,a=e._y,o=e._z,c=e._w,l=r+r,u=a+a,h=o+o,d=r*l,f=r*u,m=r*h,y=a*u,g=a*h,p=o*h,v=c*l,b=c*u,_=c*h,T=n.x,w=n.y,x=n.z;return s[0]=(1-(y+p))*T,s[1]=(f+_)*T,s[2]=(m-b)*T,s[3]=0,s[4]=(f-_)*w,s[5]=(1-(d+p))*w,s[6]=(g+v)*w,s[7]=0,s[8]=(m+b)*x,s[9]=(g-v)*x,s[10]=(1-(d+y))*x,s[11]=0,s[12]=t.x,s[13]=t.y,s[14]=t.z,s[15]=1,this}decompose(t,e,n){let s=this.elements;t.x=s[12],t.y=s[13],t.z=s[14];let r=this.determinantAffine();if(r===0)return n.set(1,1,1),e.identity(),this;let a=ts.set(s[0],s[1],s[2]).length(),o=ts.set(s[4],s[5],s[6]).length(),c=ts.set(s[8],s[9],s[10]).length();r<0&&(a=-a),yn.copy(this);let l=1/a,u=1/o,h=1/c;return yn.elements[0]*=l,yn.elements[1]*=l,yn.elements[2]*=l,yn.elements[4]*=u,yn.elements[5]*=u,yn.elements[6]*=u,yn.elements[8]*=h,yn.elements[9]*=h,yn.elements[10]*=h,e.setFromRotationMatrix(yn),n.x=a,n.y=o,n.z=c,this}makePerspective(t,e,n,s,r,a,o=gn,c=!1){let l=this.elements,u=2*r/(e-t),h=2*r/(n-s),d=(e+t)/(e-t),f=(n+s)/(n-s),m,y;if(c)m=r/(a-r),y=a*r/(a-r);else if(o===gn)m=-(a+r)/(a-r),y=-2*a*r/(a-r);else if(o===xs)m=-a/(a-r),y=-a*r/(a-r);else throw new Error("THREE.Matrix4.makePerspective(): Invalid coordinate system: "+o);return l[0]=u,l[4]=0,l[8]=d,l[12]=0,l[1]=0,l[5]=h,l[9]=f,l[13]=0,l[2]=0,l[6]=0,l[10]=m,l[14]=y,l[3]=0,l[7]=0,l[11]=-1,l[15]=0,this}makeOrthographic(t,e,n,s,r,a,o=gn,c=!1){let l=this.elements,u=2/(e-t),h=2/(n-s),d=-(e+t)/(e-t),f=-(n+s)/(n-s),m,y;if(c)m=1/(a-r),y=a/(a-r);else if(o===gn)m=-2/(a-r),y=-(a+r)/(a-r);else if(o===xs)m=-1/(a-r),y=-r/(a-r);else throw new Error("THREE.Matrix4.makeOrthographic(): Invalid coordinate system: "+o);return l[0]=u,l[4]=0,l[8]=0,l[12]=d,l[1]=0,l[5]=h,l[9]=0,l[13]=f,l[2]=0,l[6]=0,l[10]=m,l[14]=y,l[3]=0,l[7]=0,l[11]=0,l[15]=1,this}equals(t){let e=this.elements,n=t.elements;for(let s=0;s<16;s++)if(e[s]!==n[s])return!1;return!0}fromArray(t,e=0){for(let n=0;n<16;n++)this.elements[n]=t[n+e];return this}toArray(t=[],e=0){let n=this.elements;return t[e]=n[0],t[e+1]=n[1],t[e+2]=n[2],t[e+3]=n[3],t[e+4]=n[4],t[e+5]=n[5],t[e+6]=n[6],t[e+7]=n[7],t[e+8]=n[8],t[e+9]=n[9],t[e+10]=n[10],t[e+11]=n[11],t[e+12]=n[12],t[e+13]=n[13],t[e+14]=n[14],t[e+15]=n[15],t}},ts=new F,yn=new xe,Af=new F(0,0,0),Ef=new F(1,1,1),ui=new F,na=new F,rn=new F,ph=new xe,mh=new ln,jn=class i{constructor(t=0,e=0,n=0,s=i.DEFAULT_ORDER){this.isEuler=!0,this._x=t,this._y=e,this._z=n,this._order=s}get x(){return this._x}set x(t){this._x=t,this._onChangeCallback()}get y(){return this._y}set y(t){this._y=t,this._onChangeCallback()}get z(){return this._z}set z(t){this._z=t,this._onChangeCallback()}get order(){return this._order}set order(t){this._order=t,this._onChangeCallback()}set(t,e,n,s=this._order){return this._x=t,this._y=e,this._z=n,this._order=s,this._onChangeCallback(),this}clone(){return new this.constructor(this._x,this._y,this._z,this._order)}copy(t){return this._x=t._x,this._y=t._y,this._z=t._z,this._order=t._order,this._onChangeCallback(),this}setFromRotationMatrix(t,e=this._order,n=!0){let s=t.elements,r=s[0],a=s[4],o=s[8],c=s[1],l=s[5],u=s[9],h=s[2],d=s[6],f=s[10];switch(e){case"XYZ":this._y=Math.asin(ne(o,-1,1)),Math.abs(o)<.9999999?(this._x=Math.atan2(-u,f),this._z=Math.atan2(-a,r)):(this._x=Math.atan2(d,l),this._z=0);break;case"YXZ":this._x=Math.asin(-ne(u,-1,1)),Math.abs(u)<.9999999?(this._y=Math.atan2(o,f),this._z=Math.atan2(c,l)):(this._y=Math.atan2(-h,r),this._z=0);break;case"ZXY":this._x=Math.asin(ne(d,-1,1)),Math.abs(d)<.9999999?(this._y=Math.atan2(-h,f),this._z=Math.atan2(-a,l)):(this._y=0,this._z=Math.atan2(c,r));break;case"ZYX":this._y=Math.asin(-ne(h,-1,1)),Math.abs(h)<.9999999?(this._x=Math.atan2(d,f),this._z=Math.atan2(c,r)):(this._x=0,this._z=Math.atan2(-a,l));break;case"YZX":this._z=Math.asin(ne(c,-1,1)),Math.abs(c)<.9999999?(this._x=Math.atan2(-u,l),this._y=Math.atan2(-h,r)):(this._x=0,this._y=Math.atan2(o,f));break;case"XZY":this._z=Math.asin(-ne(a,-1,1)),Math.abs(a)<.9999999?(this._x=Math.atan2(d,l),this._y=Math.atan2(o,r)):(this._x=Math.atan2(-u,f),this._y=0);break;default:qt("Euler: .setFromRotationMatrix() encountered an unknown order: "+e)}return this._order=e,n===!0&&this._onChangeCallback(),this}setFromQuaternion(t,e,n){return ph.makeRotationFromQuaternion(t),this.setFromRotationMatrix(ph,e,n)}setFromVector3(t,e=this._order){return this.set(t.x,t.y,t.z,e)}reorder(t){return mh.setFromEuler(this),this.setFromQuaternion(mh,t)}equals(t){return t._x===this._x&&t._y===this._y&&t._z===this._z&&t._order===this._order}fromArray(t){return this._x=t[0],this._y=t[1],this._z=t[2],t[3]!==void 0&&(this._order=t[3]),this._onChangeCallback(),this}toArray(t=[],e=0){return t[e]=this._x,t[e+1]=this._y,t[e+2]=this._z,t[e+3]=this._order,t}_onChange(t){return this._onChangeCallback=t,this}_onChangeCallback(){}*[Symbol.iterator](){yield this._x,yield this._y,yield this._z,yield this._order}};jn.DEFAULT_ORDER="XYZ";var bs=class{constructor(){this.mask=1}set(t){this.mask=(1<<t|0)>>>0}enable(t){this.mask|=1<<t|0}enableAll(){this.mask=-1}toggle(t){this.mask^=1<<t|0}disable(t){this.mask&=~(1<<t|0)}disableAll(){this.mask=0}test(t){return(this.mask&t.mask)!==0}isEnabled(t){return(this.mask&(1<<t|0))!==0}},Cf=0,gh=new F,es=new ln,Wn=new xe,ia=new F,Hs=new F,Rf=new F,Pf=new ln,xh=new F(1,0,0),_h=new F(0,1,0),yh=new F(0,0,1),vh={type:"added"},If={type:"removed"},ns={type:"childadded",child:null},ml={type:"childremoved",child:null},Ne=class i extends Sn{constructor(){super(),this.isObject3D=!0,Object.defineProperty(this,"id",{value:Cf++}),this.uuid=Zi(),this.name="",this.type="Object3D",this.parent=null,this.children=[],this.up=i.DEFAULT_UP.clone();let t=new F,e=new jn,n=new ln,s=new F(1,1,1);function r(){n.setFromEuler(e,!1)}function a(){e.setFromQuaternion(n,void 0,!1)}e._onChange(r),n._onChange(a),Object.defineProperties(this,{position:{configurable:!0,enumerable:!0,value:t},rotation:{configurable:!0,enumerable:!0,value:e},quaternion:{configurable:!0,enumerable:!0,value:n},scale:{configurable:!0,enumerable:!0,value:s},modelViewMatrix:{value:new xe},normalMatrix:{value:new Xt}}),this.matrix=new xe,this.matrixWorld=new xe,this.matrixAutoUpdate=i.DEFAULT_MATRIX_AUTO_UPDATE,this.matrixWorldAutoUpdate=i.DEFAULT_MATRIX_WORLD_AUTO_UPDATE,this.matrixWorldNeedsUpdate=!1,this.layers=new bs,this.visible=!0,this.castShadow=!1,this.receiveShadow=!1,this.frustumCulled=!0,this.renderOrder=0,this.animations=[],this.customDepthMaterial=void 0,this.customDistanceMaterial=void 0,this.static=!1,this.userData={},this.pivot=null}onBeforeShadow(){}onAfterShadow(){}onBeforeRender(){}onAfterRender(){}applyMatrix4(t){this.matrixAutoUpdate&&this.updateMatrix(),this.matrix.premultiply(t),this.matrix.decompose(this.position,this.quaternion,this.scale),this.pivot!==null&&this.position.copy(this.pivot).applyMatrix4(this.matrix).sub(this.pivot)}applyQuaternion(t){return this.quaternion.premultiply(t),this}setRotationFromAxisAngle(t,e){this.quaternion.setFromAxisAngle(t,e)}setRotationFromEuler(t){this.quaternion.setFromEuler(t,!0)}setRotationFromMatrix(t){this.quaternion.setFromRotationMatrix(t)}setRotationFromQuaternion(t){this.quaternion.copy(t)}rotateOnAxis(t,e){return es.setFromAxisAngle(t,e),this.quaternion.multiply(es),this}rotateOnWorldAxis(t,e){return es.setFromAxisAngle(t,e),this.quaternion.premultiply(es),this}rotateX(t){return this.rotateOnAxis(xh,t)}rotateY(t){return this.rotateOnAxis(_h,t)}rotateZ(t){return this.rotateOnAxis(yh,t)}translateOnAxis(t,e){return gh.copy(t).applyQuaternion(this.quaternion),this.position.add(gh.multiplyScalar(e)),this}translateX(t){return this.translateOnAxis(xh,t)}translateY(t){return this.translateOnAxis(_h,t)}translateZ(t){return this.translateOnAxis(yh,t)}localToWorld(t){return this.updateWorldMatrix(!0,!1),t.applyMatrix4(this.matrixWorld)}worldToLocal(t){return this.updateWorldMatrix(!0,!1),t.applyMatrix4(Wn.copy(this.matrixWorld).invert())}lookAt(t,e,n){t.isVector3?ia.copy(t):ia.set(t,e,n);let s=this.parent;this.updateWorldMatrix(!0,!1),Hs.setFromMatrixPosition(this.matrixWorld),this.isCamera||this.isLight?Wn.lookAt(Hs,ia,this.up):Wn.lookAt(ia,Hs,this.up),this.quaternion.setFromRotationMatrix(Wn),s&&(Wn.extractRotation(s.matrixWorld),es.setFromRotationMatrix(Wn),this.quaternion.premultiply(es.invert()))}add(t){if(arguments.length>1){for(let e=0;e<arguments.length;e++)this.add(arguments[e]);return this}return t===this?(Zt("Object3D.add: object can't be added as a child of itself.",t),this):(t&&t.isObject3D?(t.removeFromParent(),t.parent=this,this.children.push(t),t.dispatchEvent(vh),ns.child=t,this.dispatchEvent(ns),ns.child=null):Zt("Object3D.add: object not an instance of THREE.Object3D.",t),this)}remove(t){if(arguments.length>1){for(let n=0;n<arguments.length;n++)this.remove(arguments[n]);return this}let e=this.children.indexOf(t);return e!==-1&&(t.parent=null,this.children.splice(e,1),t.dispatchEvent(If),ml.child=t,this.dispatchEvent(ml),ml.child=null),this}removeFromParent(){let t=this.parent;return t!==null&&t.remove(this),this}clear(){return this.remove(...this.children)}attach(t){return this.updateWorldMatrix(!0,!1),Wn.copy(this.matrixWorld).invert(),t.parent!==null&&(t.parent.updateWorldMatrix(!0,!1),Wn.multiply(t.parent.matrixWorld)),t.applyMatrix4(Wn),t.removeFromParent(),t.parent=this,this.children.push(t),t.updateWorldMatrix(!1,!0),t.dispatchEvent(vh),ns.child=t,this.dispatchEvent(ns),ns.child=null,this}getObjectById(t){return this.getObjectByProperty("id",t)}getObjectByName(t){return this.getObjectByProperty("name",t)}getObjectByProperty(t,e){if(this[t]===e)return this;for(let n=0,s=this.children.length;n<s;n++){let a=this.children[n].getObjectByProperty(t,e);if(a!==void 0)return a}}getObjectsByProperty(t,e,n=[]){this[t]===e&&n.push(this);let s=this.children;for(let r=0,a=s.length;r<a;r++)s[r].getObjectsByProperty(t,e,n);return n}getWorldPosition(t){return this.updateWorldMatrix(!0,!1),t.setFromMatrixPosition(this.matrixWorld)}getWorldQuaternion(t){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Hs,t,Rf),t}getWorldScale(t){return this.updateWorldMatrix(!0,!1),this.matrixWorld.decompose(Hs,Pf,t),t}getWorldDirection(t){this.updateWorldMatrix(!0,!1);let e=this.matrixWorld.elements;return t.set(e[8],e[9],e[10]).normalize()}raycast(){}intersectsFrustum(){return!0}traverse(t){t(this);let e=this.children;for(let n=0,s=e.length;n<s;n++)e[n].traverse(t)}traverseVisible(t){if(this.visible===!1)return;t(this);let e=this.children;for(let n=0,s=e.length;n<s;n++)e[n].traverseVisible(t)}traverseAncestors(t){let e=this.parent;e!==null&&(t(e),e.traverseAncestors(t))}updateMatrix(){this.matrix.compose(this.position,this.quaternion,this.scale);let t=this.pivot;if(t!==null){let e=t.x,n=t.y,s=t.z,r=this.matrix.elements;r[12]+=e-r[0]*e-r[4]*n-r[8]*s,r[13]+=n-r[1]*e-r[5]*n-r[9]*s,r[14]+=s-r[2]*e-r[6]*n-r[10]*s}this.matrixWorldNeedsUpdate=!0}updateMatrixWorld(t){this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||t)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,t=!0);let e=this.children;for(let n=0,s=e.length;n<s;n++)e[n].updateMatrixWorld(t)}updateWorldMatrix(t,e,n=!1){let s=this.parent;if(t===!0&&s!==null&&s.updateWorldMatrix(!0,!1),this.matrixAutoUpdate&&this.updateMatrix(),(this.matrixWorldNeedsUpdate||n)&&(this.matrixWorldAutoUpdate===!0&&(this.parent===null?this.matrixWorld.copy(this.matrix):this.matrixWorld.multiplyMatrices(this.parent.matrixWorld,this.matrix)),this.matrixWorldNeedsUpdate=!1,n=!0),e===!0){let r=this.children;for(let a=0,o=r.length;a<o;a++)r[a].updateWorldMatrix(!1,!0,n)}}toJSON(t){let e=t===void 0||typeof t=="string",n={};e&&(t={geometries:{},materials:{},textures:{},images:{},shapes:{},skeletons:{},animations:{},nodes:{}},n.metadata={version:4.7,type:"Object",generator:"Object3D.toJSON"});let s={};s.uuid=this.uuid,s.type=this.type,s.name=this.name,s.castShadow=this.castShadow,s.receiveShadow=this.receiveShadow,s.visible=this.visible,s.frustumCulled=this.frustumCulled,s.renderOrder=this.renderOrder,s.static=this.static,s.matrixAutoUpdate=this.matrixAutoUpdate,Object.keys(this.userData).length>0&&(s.userData=this.userData),s.layers=this.layers.mask,s.matrix=this.matrix.toArray(),s.up=this.up.toArray(),this.pivot!==null&&(s.pivot=this.pivot.toArray()),this.morphTargetDictionary!==void 0&&(s.morphTargetDictionary=Object.assign({},this.morphTargetDictionary)),this.morphTargetInfluences!==void 0&&(s.morphTargetInfluences=this.morphTargetInfluences.slice()),this.isInstancedMesh&&(s.type="InstancedMesh",s.count=this.count,s.instanceMatrix=this.instanceMatrix.toJSON(),this.instanceColor!==null&&(s.instanceColor=this.instanceColor.toJSON())),this.isBatchedMesh&&(s.type="BatchedMesh",s.perObjectFrustumCulled=this.perObjectFrustumCulled,s.sortObjects=this.sortObjects,s.drawRanges=this._drawRanges,s.reservedRanges=this._reservedRanges,s.geometryInfo=this._geometryInfo.map(o=>({...o,boundingBox:o.boundingBox?o.boundingBox.toJSON():void 0,boundingSphere:o.boundingSphere?o.boundingSphere.toJSON():void 0})),s.instanceInfo=this._instanceInfo.map(o=>({...o})),s.availableInstanceIds=this._availableInstanceIds.slice(),s.availableGeometryIds=this._availableGeometryIds.slice(),s.nextIndexStart=this._nextIndexStart,s.nextVertexStart=this._nextVertexStart,s.geometryCount=this._geometryCount,s.maxInstanceCount=this._maxInstanceCount,s.maxVertexCount=this._maxVertexCount,s.maxIndexCount=this._maxIndexCount,s.geometryInitialized=this._geometryInitialized,s.matricesTexture=this._matricesTexture.toJSON(t),s.indirectTexture=this._indirectTexture.toJSON(t),this._colorsTexture!==null&&(s.colorsTexture=this._colorsTexture.toJSON(t)),this.boundingSphere!==null&&(s.boundingSphere=this.boundingSphere.toJSON()),this.boundingBox!==null&&(s.boundingBox=this.boundingBox.toJSON()));function r(o,c){return o[c.uuid]===void 0&&(o[c.uuid]=c.toJSON(t)),c.uuid}if(this.isScene)this.background&&(this.background.isColor?s.background=this.background.toJSON():this.background.isTexture&&(s.background=this.background.toJSON(t).uuid)),this.environment&&this.environment.isTexture&&this.environment.isRenderTargetTexture!==!0&&(s.environment=this.environment.toJSON(t).uuid);else if(this.isMesh||this.isLine||this.isPoints){s.geometry=r(t.geometries,this.geometry);let o=this.geometry.parameters;if(o!==void 0&&o.shapes!==void 0){let c=o.shapes;if(Array.isArray(c))for(let l=0,u=c.length;l<u;l++){let h=c[l];r(t.shapes,h)}else r(t.shapes,c)}}if(this.isSkinnedMesh&&(s.bindMode=this.bindMode,s.bindMatrix=this.bindMatrix.toArray(),this.skeleton!==void 0&&(r(t.skeletons,this.skeleton),s.skeleton=this.skeleton.uuid)),this.material!==void 0)if(Array.isArray(this.material)){let o=[];for(let c=0,l=this.material.length;c<l;c++)o.push(r(t.materials,this.material[c]));s.material=o}else s.material=r(t.materials,this.material);if(this.children.length>0){s.children=[];for(let o=0;o<this.children.length;o++)s.children.push(this.children[o].toJSON(t).object)}if(this.animations.length>0){s.animations=[];for(let o=0;o<this.animations.length;o++){let c=this.animations[o];s.animations.push(r(t.animations,c))}}if(e){let o=a(t.geometries),c=a(t.materials),l=a(t.textures),u=a(t.images),h=a(t.shapes),d=a(t.skeletons),f=a(t.animations),m=a(t.nodes);o.length>0&&(n.geometries=o),c.length>0&&(n.materials=c),l.length>0&&(n.textures=l),u.length>0&&(n.images=u),h.length>0&&(n.shapes=h),d.length>0&&(n.skeletons=d),f.length>0&&(n.animations=f),m.length>0&&(n.nodes=m)}return n.object=s,n;function a(o){let c=[];for(let l in o){let u=o[l];delete u.metadata,c.push(u)}return c}}clone(t){return new this.constructor().copy(this,t)}copy(t,e=!0){if(this.name=t.name,this.up.copy(t.up),this.position.copy(t.position),this.rotation.order=t.rotation.order,this.quaternion.copy(t.quaternion),this.scale.copy(t.scale),this.pivot=t.pivot!==null?t.pivot.clone():null,this.matrix.copy(t.matrix),this.matrixWorld.copy(t.matrixWorld),this.matrixAutoUpdate=t.matrixAutoUpdate,this.matrixWorldAutoUpdate=t.matrixWorldAutoUpdate,this.matrixWorldNeedsUpdate=t.matrixWorldNeedsUpdate,this.layers.mask=t.layers.mask,this.visible=t.visible,this.castShadow=t.castShadow,this.receiveShadow=t.receiveShadow,this.frustumCulled=t.frustumCulled,this.renderOrder=t.renderOrder,this.static=t.static,this.animations=t.animations.slice(),this.userData=JSON.parse(JSON.stringify(t.userData)),e===!0)for(let n=0;n<t.children.length;n++){let s=t.children[n];this.add(s.clone())}return this}dispose(){this.dispatchEvent({type:"dispose"})}};Ne.DEFAULT_UP=new F(0,1,0);Ne.DEFAULT_MATRIX_AUTO_UPDATE=!0;Ne.DEFAULT_MATRIX_WORLD_AUTO_UPDATE=!0;var ce=class extends Ne{constructor(){super(),this.isGroup=!0,this.type="Group"}},Lf={type:"move"},Ms=class{constructor(){this._targetRay=null,this._grip=null,this._hand=null}getHandSpace(){return this._hand===null&&(this._hand=new ce,this._hand.matrixAutoUpdate=!1,this._hand.visible=!1,this._hand.joints={},this._hand.inputState={pinching:!1}),this._hand}getTargetRaySpace(){return this._targetRay===null&&(this._targetRay=new ce,this._targetRay.matrixAutoUpdate=!1,this._targetRay.visible=!1,this._targetRay.hasLinearVelocity=!1,this._targetRay.linearVelocity=new F,this._targetRay.hasAngularVelocity=!1,this._targetRay.angularVelocity=new F),this._targetRay}getGripSpace(){return this._grip===null&&(this._grip=new ce,this._grip.matrixAutoUpdate=!1,this._grip.visible=!1,this._grip.hasLinearVelocity=!1,this._grip.linearVelocity=new F,this._grip.hasAngularVelocity=!1,this._grip.angularVelocity=new F,this._grip.eventsEnabled=!1),this._grip}dispatchEvent(t){return this._targetRay!==null&&this._targetRay.dispatchEvent(t),this._grip!==null&&this._grip.dispatchEvent(t),this._hand!==null&&this._hand.dispatchEvent(t),this}connect(t){if(t&&t.hand){let e=this._hand;if(e)for(let n of t.hand.values())this._getHandJoint(e,n)}return this.dispatchEvent({type:"connected",data:t}),this}disconnect(t){return this.dispatchEvent({type:"disconnected",data:t}),this._targetRay!==null&&(this._targetRay.visible=!1),this._grip!==null&&(this._grip.visible=!1),this._hand!==null&&(this._hand.visible=!1),this}update(t,e,n){let s=null,r=null,a=null,o=this._targetRay,c=this._grip,l=this._hand;if(t&&e.session.visibilityState!=="visible-blurred"){if(l&&t.hand){a=!0;for(let y of t.hand.values()){let g=e.getJointPose(y,n),p=this._getHandJoint(l,y);g!==null&&(p.matrix.fromArray(g.transform.matrix),p.matrix.decompose(p.position,p.quaternion,p.scale),p.matrixWorldNeedsUpdate=!0,p.jointRadius=g.radius),p.visible=g!==null}let u=l.joints["index-finger-tip"],h=l.joints["thumb-tip"],d=u.position.distanceTo(h.position),f=.02,m=.005;l.inputState.pinching&&d>f+m?(l.inputState.pinching=!1,this.dispatchEvent({type:"pinchend",handedness:t.handedness,target:this})):!l.inputState.pinching&&d<=f-m&&(l.inputState.pinching=!0,this.dispatchEvent({type:"pinchstart",handedness:t.handedness,target:this}))}else c!==null&&t.gripSpace&&(r=e.getPose(t.gripSpace,n),r!==null&&(c.matrix.fromArray(r.transform.matrix),c.matrix.decompose(c.position,c.quaternion,c.scale),c.matrixWorldNeedsUpdate=!0,r.linearVelocity?(c.hasLinearVelocity=!0,c.linearVelocity.copy(r.linearVelocity)):c.hasLinearVelocity=!1,r.angularVelocity?(c.hasAngularVelocity=!0,c.angularVelocity.copy(r.angularVelocity)):c.hasAngularVelocity=!1,c.eventsEnabled&&c.dispatchEvent({type:"gripUpdated",data:t,target:this})));o!==null&&(s=e.getPose(t.targetRaySpace,n),s===null&&r!==null&&(s=r),s!==null&&(o.matrix.fromArray(s.transform.matrix),o.matrix.decompose(o.position,o.quaternion,o.scale),o.matrixWorldNeedsUpdate=!0,s.linearVelocity?(o.hasLinearVelocity=!0,o.linearVelocity.copy(s.linearVelocity)):o.hasLinearVelocity=!1,s.angularVelocity?(o.hasAngularVelocity=!0,o.angularVelocity.copy(s.angularVelocity)):o.hasAngularVelocity=!1,this.dispatchEvent(Lf)))}return o!==null&&(o.visible=s!==null),c!==null&&(c.visible=r!==null),l!==null&&(l.visible=a!==null),this}_getHandJoint(t,e){if(t.joints[e.jointName]===void 0){let n=new ce;n.matrixAutoUpdate=!1,n.visible=!1,t.joints[e.jointName]=n,t.add(n)}return t.joints[e.jointName]}},Du={aliceblue:15792383,antiquewhite:16444375,aqua:65535,aquamarine:8388564,azure:15794175,beige:16119260,bisque:16770244,black:0,blanchedalmond:16772045,blue:255,blueviolet:9055202,brown:10824234,burlywood:14596231,cadetblue:6266528,chartreuse:8388352,chocolate:13789470,coral:16744272,cornflowerblue:6591981,cornsilk:16775388,crimson:14423100,cyan:65535,darkblue:139,darkcyan:35723,darkgoldenrod:12092939,darkgray:11119017,darkgreen:25600,darkgrey:11119017,darkkhaki:12433259,darkmagenta:9109643,darkolivegreen:5597999,darkorange:16747520,darkorchid:10040012,darkred:9109504,darksalmon:15308410,darkseagreen:9419919,darkslateblue:4734347,darkslategray:3100495,darkslategrey:3100495,darkturquoise:52945,darkviolet:9699539,deeppink:16716947,deepskyblue:49151,dimgray:6908265,dimgrey:6908265,dodgerblue:2003199,firebrick:11674146,floralwhite:16775920,forestgreen:2263842,fuchsia:16711935,gainsboro:14474460,ghostwhite:16316671,gold:16766720,goldenrod:14329120,gray:8421504,green:32768,greenyellow:11403055,grey:8421504,honeydew:15794160,hotpink:16738740,indianred:13458524,indigo:4915330,ivory:16777200,khaki:15787660,lavender:15132410,lavenderblush:16773365,lawngreen:8190976,lemonchiffon:16775885,lightblue:11393254,lightcoral:15761536,lightcyan:14745599,lightgoldenrodyellow:16448210,lightgray:13882323,lightgreen:9498256,lightgrey:13882323,lightpink:16758465,lightsalmon:16752762,lightseagreen:2142890,lightskyblue:8900346,lightslategray:7833753,lightslategrey:7833753,lightsteelblue:11584734,lightyellow:16777184,lime:65280,limegreen:3329330,linen:16445670,magenta:16711935,maroon:8388608,mediumaquamarine:6737322,mediumblue:205,mediumorchid:12211667,mediumpurple:9662683,mediumseagreen:3978097,mediumslateblue:8087790,mediumspringgreen:64154,mediumturquoise:4772300,mediumvioletred:13047173,midnightblue:1644912,mintcream:16121850,mistyrose:16770273,moccasin:16770229,navajowhite:16768685,navy:128,oldlace:16643558,olive:8421376,olivedrab:7048739,orange:16753920,orangered:16729344,orchid:14315734,palegoldenrod:15657130,palegreen:10025880,paleturquoise:11529966,palevioletred:14381203,papayawhip:16773077,peachpuff:16767673,peru:13468991,pink:16761035,plum:14524637,powderblue:11591910,purple:8388736,rebeccapurple:6697881,red:16711680,rosybrown:12357519,royalblue:4286945,saddlebrown:9127187,salmon:16416882,sandybrown:16032864,seagreen:3050327,seashell:16774638,sienna:10506797,silver:12632256,skyblue:8900331,slateblue:6970061,slategray:7372944,slategrey:7372944,snow:16775930,springgreen:65407,steelblue:4620980,tan:13808780,teal:32896,thistle:14204888,tomato:16737095,turquoise:4251856,violet:15631086,wheat:16113331,white:16777215,whitesmoke:16119285,yellow:16776960,yellowgreen:10145074},di={h:0,s:0,l:0},sa={h:0,s:0,l:0};function gl(i,t,e){return e<0&&(e+=1),e>1&&(e-=1),e<1/6?i+(t-i)*6*e:e<1/2?t:e<2/3?i+(t-i)*6*(2/3-e):i}var Vt=class{constructor(t,e,n){return this.isColor=!0,this.r=1,this.g=1,this.b=1,this.set(t,e,n)}set(t,e,n){if(e===void 0&&n===void 0){let s=t;s&&s.isColor?this.copy(s):typeof s=="number"?this.setHex(s):typeof s=="string"&&this.setStyle(s)}else this.setRGB(t,e,n);return this}setScalar(t){return this.r=t,this.g=t,this.b=t,this}setHex(t,e=He){return t=Math.floor(t),this.r=(t>>16&255)/255,this.g=(t>>8&255)/255,this.b=(t&255)/255,oe.colorSpaceToWorking(this,e),this}setRGB(t,e,n,s=oe.workingColorSpace){return this.r=t,this.g=e,this.b=n,oe.colorSpaceToWorking(this,s),this}setHSL(t,e,n,s=oe.workingColorSpace){if(t=vc(t,1),e=ne(e,0,1),n=ne(n,0,1),e===0)this.r=this.g=this.b=n;else{let r=n<=.5?n*(1+e):n+e-n*e,a=2*n-r;this.r=gl(a,r,t+1/3),this.g=gl(a,r,t),this.b=gl(a,r,t-1/3)}return oe.colorSpaceToWorking(this,s),this}setStyle(t,e=He){function n(r){r!==void 0&&parseFloat(r)<1&&qt("Color: Alpha component of "+t+" will be ignored.")}let s;if(s=/^(\w+)\(([^\)]*)\)/.exec(t)){let r,a=s[1],o=s[2];switch(a){case"rgb":case"rgba":if(r=/^\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*(?:,\s*(\d+(?:\.\d+)?|\.\d+)\s*)?$/.exec(o))return n(r[4]),this.setRGB(Math.min(255,parseInt(r[1],10))/255,Math.min(255,parseInt(r[2],10))/255,Math.min(255,parseInt(r[3],10))/255,e);if(r=/^\s*(\d+(?:\.\d+)?|\.\d+)\%\s*,\s*(\d+(?:\.\d+)?|\.\d+)\%\s*,\s*(\d+(?:\.\d+)?|\.\d+)\%\s*(?:,\s*(\d+(?:\.\d+)?|\.\d+)\s*)?$/.exec(o))return n(r[4]),this.setRGB(Math.min(100,parseFloat(r[1]))/100,Math.min(100,parseFloat(r[2]))/100,Math.min(100,parseFloat(r[3]))/100,e);break;case"hsl":case"hsla":if(r=/^\s*([+-]?(?:\d+(?:\.\d+)?|\.\d+))\s*,\s*(\d+(?:\.\d+)?|\.\d+)\%\s*,\s*(\d+(?:\.\d+)?|\.\d+)\%\s*(?:,\s*(\d+(?:\.\d+)?|\.\d+)\s*)?$/.exec(o))return n(r[4]),this.setHSL(parseFloat(r[1])/360,parseFloat(r[2])/100,parseFloat(r[3])/100,e);break;default:qt("Color: Unknown color model "+t)}}else if(s=/^\#([A-Fa-f\d]+)$/.exec(t)){let r=s[1],a=r.length;if(a===3)return this.setRGB(parseInt(r.charAt(0),16)/15,parseInt(r.charAt(1),16)/15,parseInt(r.charAt(2),16)/15,e);if(a===6)return this.setHex(parseInt(r,16),e);qt("Color: Invalid hex color "+t)}else if(t&&t.length>0)return this.setColorName(t,e);return this}setColorName(t,e=He){let n=Du[t.toLowerCase()];return n!==void 0?this.setHex(n,e):qt("Color: Unknown color "+t),this}clone(){return new this.constructor(this.r,this.g,this.b)}copy(t){return this.r=t.r,this.g=t.g,this.b=t.b,this}copySRGBToLinear(t){return this.r=Jn(t.r),this.g=Jn(t.g),this.b=Jn(t.b),this}copyLinearToSRGB(t){return this.r=fs(t.r),this.g=fs(t.g),this.b=fs(t.b),this}convertSRGBToLinear(){return this.copySRGBToLinear(this),this}convertLinearToSRGB(){return this.copyLinearToSRGB(this),this}getHex(t=He){return oe.workingToColorSpace($e.copy(this),t),Math.round(ne($e.r*255,0,255))*65536+Math.round(ne($e.g*255,0,255))*256+Math.round(ne($e.b*255,0,255))}getHexString(t=He){return("000000"+this.getHex(t).toString(16)).slice(-6)}getHSL(t,e=oe.workingColorSpace){oe.workingToColorSpace($e.copy(this),e);let n=$e.r,s=$e.g,r=$e.b,a=Math.max(n,s,r),o=Math.min(n,s,r),c,l,u=(o+a)/2;if(o===a)c=0,l=0;else{let h=a-o;switch(l=u<=.5?h/(a+o):h/(2-a-o),a){case n:c=(s-r)/h+(s<r?6:0);break;case s:c=(r-n)/h+2;break;case r:c=(n-s)/h+4;break}c/=6}return t.h=c,t.s=l,t.l=u,t}getRGB(t,e=oe.workingColorSpace){return oe.workingToColorSpace($e.copy(this),e),t.r=$e.r,t.g=$e.g,t.b=$e.b,t}getStyle(t=He){oe.workingToColorSpace($e.copy(this),t);let e=$e.r,n=$e.g,s=$e.b;return t!==He?`color(${t} ${e.toFixed(3)} ${n.toFixed(3)} ${s.toFixed(3)})`:`rgb(${Math.round(e*255)},${Math.round(n*255)},${Math.round(s*255)})`}offsetHSL(t,e,n){return this.getHSL(di),this.setHSL(di.h+t,di.s+e,di.l+n)}add(t){return this.r+=t.r,this.g+=t.g,this.b+=t.b,this}addColors(t,e){return this.r=t.r+e.r,this.g=t.g+e.g,this.b=t.b+e.b,this}addScalar(t){return this.r+=t,this.g+=t,this.b+=t,this}sub(t){return this.r=Math.max(0,this.r-t.r),this.g=Math.max(0,this.g-t.g),this.b=Math.max(0,this.b-t.b),this}multiply(t){return this.r*=t.r,this.g*=t.g,this.b*=t.b,this}multiplyScalar(t){return this.r*=t,this.g*=t,this.b*=t,this}lerp(t,e){return this.r+=(t.r-this.r)*e,this.g+=(t.g-this.g)*e,this.b+=(t.b-this.b)*e,this}lerpColors(t,e,n){return this.r=t.r+(e.r-t.r)*n,this.g=t.g+(e.g-t.g)*n,this.b=t.b+(e.b-t.b)*n,this}lerpHSL(t,e){this.getHSL(di),t.getHSL(sa);let n=Qs(di.h,sa.h,e),s=Qs(di.s,sa.s,e),r=Qs(di.l,sa.l,e);return this.setHSL(n,s,r),this}setFromVector3(t){return this.r=t.x,this.g=t.y,this.b=t.z,this}applyMatrix3(t){let e=this.r,n=this.g,s=this.b,r=t.elements;return this.r=r[0]*e+r[3]*n+r[6]*s,this.g=r[1]*e+r[4]*n+r[7]*s,this.b=r[2]*e+r[5]*n+r[8]*s,this}equals(t){return t.r===this.r&&t.g===this.g&&t.b===this.b}fromArray(t,e=0){return this.r=t[e],this.g=t[e+1],this.b=t[e+2],this}toArray(t=[],e=0){return t[e]=this.r,t[e+1]=this.g,t[e+2]=this.b,t}fromBufferAttribute(t,e){return this.r=t.getX(e),this.g=t.getY(e),this.b=t.getZ(e),this}toJSON(){return this.getHex()}*[Symbol.iterator](){yield this.r,yield this.g,yield this.b}},$e=new Vt;Vt.NAMES=Du;var or=class i{constructor(t,e=25e-5){this.isFogExp2=!0,this.name="",this.color=new Vt(t),this.density=e}clone(){return new i(this.color,this.density)}toJSON(){return{type:"FogExp2",name:this.name,color:this.color.getHex(),density:this.density}}};var wn=class extends Ne{constructor(){super(),this.isScene=!0,this.type="Scene",this.background=null,this.environment=null,this.fog=null,this.backgroundBlurriness=0,this.backgroundIntensity=1,this.backgroundRotation=new jn,this.environmentIntensity=1,this.environmentRotation=new jn,this.overrideMaterial=null,typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}copy(t,e){return super.copy(t,e),this.background=t.background!==null?t.background.clone():null,this.environment=t.environment!==null?t.environment.clone():null,this.fog=t.fog!==null?t.fog.clone():null,this.backgroundBlurriness=t.backgroundBlurriness,this.backgroundIntensity=t.backgroundIntensity,this.backgroundRotation.copy(t.backgroundRotation),this.environmentIntensity=t.environmentIntensity,this.environmentRotation.copy(t.environmentRotation),this.overrideMaterial=t.overrideMaterial!==null?t.overrideMaterial.clone():null,this.matrixAutoUpdate=t.matrixAutoUpdate,this}toJSON(t){let e=super.toJSON(t);return this.fog!==null&&(e.object.fog=this.fog.toJSON()),e.object.backgroundBlurriness=this.backgroundBlurriness,e.object.backgroundIntensity=this.backgroundIntensity,e.object.backgroundRotation=this.backgroundRotation.toArray(),e.object.environmentIntensity=this.environmentIntensity,e.object.environmentRotation=this.environmentRotation.toArray(),e}},vn=new F,Xn=new F,xl=new F,qn=new F,is=new F,ss=new F,bh=new F,_l=new F,yl=new F,vl=new F,bl=new Se,Ml=new Se,Sl=new Se,gi=class i{constructor(t=new F,e=new F,n=new F){this.a=t,this.b=e,this.c=n}static getNormal(t,e,n,s){s.subVectors(n,e),vn.subVectors(t,e),s.cross(vn);let r=s.lengthSq();return r>0?s.multiplyScalar(1/Math.sqrt(r)):s.set(0,0,0)}static getBarycoord(t,e,n,s,r){vn.subVectors(s,e),Xn.subVectors(n,e),xl.subVectors(t,e);let a=vn.dot(vn),o=vn.dot(Xn),c=vn.dot(xl),l=Xn.dot(Xn),u=Xn.dot(xl),h=a*l-o*o;if(h===0)return r.set(0,0,0),null;let d=1/h,f=(l*c-o*u)*d,m=(a*u-o*c)*d;return r.set(1-f-m,m,f)}static containsPoint(t,e,n,s){return this.getBarycoord(t,e,n,s,qn)===null?!1:qn.x>=0&&qn.y>=0&&qn.x+qn.y<=1}static getInterpolation(t,e,n,s,r,a,o,c){return this.getBarycoord(t,e,n,s,qn)===null?(c.x=0,c.y=0,"z"in c&&(c.z=0),"w"in c&&(c.w=0),null):(c.setScalar(0),c.addScaledVector(r,qn.x),c.addScaledVector(a,qn.y),c.addScaledVector(o,qn.z),c)}static getInterpolatedAttribute(t,e,n,s,r,a){return bl.setScalar(0),Ml.setScalar(0),Sl.setScalar(0),bl.fromBufferAttribute(t,e),Ml.fromBufferAttribute(t,n),Sl.fromBufferAttribute(t,s),a.setScalar(0),a.addScaledVector(bl,r.x),a.addScaledVector(Ml,r.y),a.addScaledVector(Sl,r.z),a}static isFrontFacing(t,e,n,s){return vn.subVectors(n,e),Xn.subVectors(t,e),vn.cross(Xn).dot(s)<0}set(t,e,n){return this.a.copy(t),this.b.copy(e),this.c.copy(n),this}setFromPointsAndIndices(t,e,n,s){return this.a.copy(t[e]),this.b.copy(t[n]),this.c.copy(t[s]),this}setFromAttributeAndIndices(t,e,n,s){return this.a.fromBufferAttribute(t,e),this.b.fromBufferAttribute(t,n),this.c.fromBufferAttribute(t,s),this}clone(){return new this.constructor().copy(this)}copy(t){return this.a.copy(t.a),this.b.copy(t.b),this.c.copy(t.c),this}getArea(){return vn.subVectors(this.c,this.b),Xn.subVectors(this.a,this.b),vn.cross(Xn).length()*.5}getMidpoint(t){return t.addVectors(this.a,this.b).add(this.c).multiplyScalar(1/3)}getNormal(t){return i.getNormal(this.a,this.b,this.c,t)}getPlane(t){return t.setFromCoplanarPoints(this.a,this.b,this.c)}getBarycoord(t,e){return i.getBarycoord(t,this.a,this.b,this.c,e)}getInterpolation(t,e,n,s,r){return i.getInterpolation(t,this.a,this.b,this.c,e,n,s,r)}containsPoint(t){return i.containsPoint(t,this.a,this.b,this.c)}isFrontFacing(t){return i.isFrontFacing(this.a,this.b,this.c,t)}intersectsBox(t){return t.intersectsTriangle(this)}closestPointToPoint(t,e){let n=this.a,s=this.b,r=this.c,a,o;is.subVectors(s,n),ss.subVectors(r,n),_l.subVectors(t,n);let c=is.dot(_l),l=ss.dot(_l);if(c<=0&&l<=0)return e.copy(n);yl.subVectors(t,s);let u=is.dot(yl),h=ss.dot(yl);if(u>=0&&h<=u)return e.copy(s);let d=c*h-u*l;if(d<=0&&c>=0&&u<=0&&c-u>0)return a=c/(c-u),e.copy(n).addScaledVector(is,a);vl.subVectors(t,r);let f=is.dot(vl),m=ss.dot(vl);if(m>=0&&f<=m)return e.copy(r);let y=f*l-c*m;if(y<=0&&l>=0&&m<=0)return o=l/(l-m),e.copy(n).addScaledVector(ss,o);let g=u*m-f*h;if(g<=0&&h-u>=0&&f-m>=0)return bh.subVectors(r,s),o=(h-u)/(h-u+(f-m)),e.copy(s).addScaledVector(bh,o);let p=1/(g+y+d);return a=y*p,o=d*p,e.copy(n).addScaledVector(is,a).addScaledVector(ss,o)}equals(t){return t.a.equals(this.a)&&t.b.equals(this.b)&&t.c.equals(this.c)}},Fn=class{constructor(t=new F(1/0,1/0,1/0),e=new F(-1/0,-1/0,-1/0)){this.isBox3=!0,this.min=t,this.max=e}set(t,e){return this.min.copy(t),this.max.copy(e),this}setFromArray(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e+=3)this.expandByPoint(bn.fromArray(t,e));return this}setFromBufferAttribute(t){this.makeEmpty();for(let e=0,n=t.count;e<n;e++)this.expandByPoint(bn.fromBufferAttribute(t,e));return this}setFromPoints(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e++)this.expandByPoint(t[e]);return this}setFromCenterAndSize(t,e){let n=bn.copy(e).multiplyScalar(.5);return this.min.copy(t).sub(n),this.max.copy(t).add(n),this}setFromObject(t,e=!1){return this.makeEmpty(),this.expandByObject(t,e)}clone(){return new this.constructor().copy(this)}copy(t){return this.min.copy(t.min),this.max.copy(t.max),this}makeEmpty(){return this.min.x=this.min.y=this.min.z=1/0,this.max.x=this.max.y=this.max.z=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y||this.max.z<this.min.z}getCenter(t){return this.isEmpty()?t.set(0,0,0):t.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(t){return this.isEmpty()?t.set(0,0,0):t.subVectors(this.max,this.min)}expandByPoint(t){return this.min.min(t),this.max.max(t),this}expandByVector(t){return this.min.sub(t),this.max.add(t),this}expandByScalar(t){return this.min.addScalar(-t),this.max.addScalar(t),this}expandByObject(t,e=!1){t.updateWorldMatrix(!1,!1);let n=t.geometry;if(n!==void 0){let r=n.getAttribute("position");if(e===!0&&r!==void 0&&t.isInstancedMesh!==!0&&t.isBatchedMesh!==!0)for(let a=0,o=r.count;a<o;a++)t.isMesh===!0?t.getVertexPosition(a,bn):bn.fromBufferAttribute(r,a),bn.applyMatrix4(t.matrixWorld),this.expandByPoint(bn);else t.boundingBox!==void 0?(t.boundingBox===null&&t.computeBoundingBox(),ra.copy(t.boundingBox)):(n.boundingBox===null&&n.computeBoundingBox(),ra.copy(n.boundingBox)),ra.applyMatrix4(t.matrixWorld),this.union(ra)}let s=t.children;for(let r=0,a=s.length;r<a;r++)this.expandByObject(s[r],e);return this}containsPoint(t){return t.x>=this.min.x&&t.x<=this.max.x&&t.y>=this.min.y&&t.y<=this.max.y&&t.z>=this.min.z&&t.z<=this.max.z}containsBox(t){return this.min.x<=t.min.x&&t.max.x<=this.max.x&&this.min.y<=t.min.y&&t.max.y<=this.max.y&&this.min.z<=t.min.z&&t.max.z<=this.max.z}getParameter(t,e){return e.set((t.x-this.min.x)/(this.max.x-this.min.x),(t.y-this.min.y)/(this.max.y-this.min.y),(t.z-this.min.z)/(this.max.z-this.min.z))}intersectsBox(t){return t.max.x>=this.min.x&&t.min.x<=this.max.x&&t.max.y>=this.min.y&&t.min.y<=this.max.y&&t.max.z>=this.min.z&&t.min.z<=this.max.z}intersectsSphere(t){return this.clampPoint(t.center,bn),bn.distanceToSquared(t.center)<=t.radius*t.radius}intersectsPlane(t){let e,n;return t.normal.x>0?(e=t.normal.x*this.min.x,n=t.normal.x*this.max.x):(e=t.normal.x*this.max.x,n=t.normal.x*this.min.x),t.normal.y>0?(e+=t.normal.y*this.min.y,n+=t.normal.y*this.max.y):(e+=t.normal.y*this.max.y,n+=t.normal.y*this.min.y),t.normal.z>0?(e+=t.normal.z*this.min.z,n+=t.normal.z*this.max.z):(e+=t.normal.z*this.max.z,n+=t.normal.z*this.min.z),e<=-t.constant&&n>=-t.constant}intersectsTriangle(t){if(this.isEmpty())return!1;this.getCenter(Ws),aa.subVectors(this.max,Ws),rs.subVectors(t.a,Ws),as.subVectors(t.b,Ws),os.subVectors(t.c,Ws),fi.subVectors(as,rs),pi.subVectors(os,as),Li.subVectors(rs,os);let e=[0,-fi.z,fi.y,0,-pi.z,pi.y,0,-Li.z,Li.y,fi.z,0,-fi.x,pi.z,0,-pi.x,Li.z,0,-Li.x,-fi.y,fi.x,0,-pi.y,pi.x,0,-Li.y,Li.x,0];return!wl(e,rs,as,os,aa)||(e=[1,0,0,0,1,0,0,0,1],!wl(e,rs,as,os,aa))?!1:(oa.crossVectors(fi,pi),e=[oa.x,oa.y,oa.z],wl(e,rs,as,os,aa))}clampPoint(t,e){return e.copy(t).clamp(this.min,this.max)}distanceToPoint(t){return this.clampPoint(t,bn).distanceTo(t)}getBoundingSphere(t){return this.isEmpty()?t.makeEmpty():(this.getCenter(t.center),t.radius=this.getSize(bn).length()*.5),t}intersect(t){return this.min.max(t.min),this.max.min(t.max),this.isEmpty()&&this.makeEmpty(),this}union(t){return this.min.min(t.min),this.max.max(t.max),this}applyMatrix4(t){return this.isEmpty()?this:(Yn[0].set(this.min.x,this.min.y,this.min.z).applyMatrix4(t),Yn[1].set(this.min.x,this.min.y,this.max.z).applyMatrix4(t),Yn[2].set(this.min.x,this.max.y,this.min.z).applyMatrix4(t),Yn[3].set(this.min.x,this.max.y,this.max.z).applyMatrix4(t),Yn[4].set(this.max.x,this.min.y,this.min.z).applyMatrix4(t),Yn[5].set(this.max.x,this.min.y,this.max.z).applyMatrix4(t),Yn[6].set(this.max.x,this.max.y,this.min.z).applyMatrix4(t),Yn[7].set(this.max.x,this.max.y,this.max.z).applyMatrix4(t),this.setFromPoints(Yn),this)}translate(t){return this.min.add(t),this.max.add(t),this}equals(t){return t.min.equals(this.min)&&t.max.equals(this.max)}toJSON(){return{min:this.min.toArray(),max:this.max.toArray()}}fromJSON(t){return this.min.fromArray(t.min),this.max.fromArray(t.max),this}},Yn=[new F,new F,new F,new F,new F,new F,new F,new F],bn=new F,ra=new Fn,rs=new F,as=new F,os=new F,fi=new F,pi=new F,Li=new F,Ws=new F,aa=new F,oa=new F,Di=new F;function wl(i,t,e,n,s){for(let r=0,a=i.length-3;r<=a;r+=3){Di.fromArray(i,r);let o=s.x*Math.abs(Di.x)+s.y*Math.abs(Di.y)+s.z*Math.abs(Di.z),c=t.dot(Di),l=e.dot(Di),u=n.dot(Di);if(Math.max(-Math.max(c,l,u),Math.min(c,l,u))>o)return!1}return!0}var Ie=new F,la=new ft,Df=0,Mn=class extends Sn{constructor(t,e,n=!1){if(super(),Array.isArray(t))throw new TypeError("THREE.BufferAttribute: array should be a Typed Array.");this.isBufferAttribute=!0,Object.defineProperty(this,"id",{value:Df++}),this.name="",this.array=t,this.itemSize=e,this.count=t!==void 0?t.length/e:0,this.normalized=n,this.usage=Au,this.updateRanges=[],this.gpuType=xn,this.version=0,this._normalizationScale=t!==void 0?Mc(t):void 0}onUploadCallback(){}set needsUpdate(t){t===!0&&this.version++}setUsage(t){return this.usage=t,this}addUpdateRange(t,e){this.updateRanges.push({start:t,count:e})}clearUpdateRanges(){this.updateRanges.length=0}copy(t){return this.name=t.name,this.array=new t.array.constructor(t.array),this.itemSize=t.itemSize,this.count=t.count,this.normalized=t.normalized,this.usage=t.usage,this.gpuType=t.gpuType,this._normalizationScale=t._normalizationScale,this}copyAt(t,e,n){t*=this.itemSize,n*=e.itemSize;for(let s=0,r=this.itemSize;s<r;s++)this.array[t+s]=e.array[n+s];return this}copyArray(t){return this.array.set(t),this}applyMatrix3(t){if(this.itemSize===2)for(let e=0,n=this.count;e<n;e++)la.fromBufferAttribute(this,e),la.applyMatrix3(t),this.setXY(e,la.x,la.y);else if(this.itemSize===3)for(let e=0,n=this.count;e<n;e++)Ie.fromBufferAttribute(this,e),Ie.applyMatrix3(t),this.setXYZ(e,Ie.x,Ie.y,Ie.z);return this}applyMatrix4(t){for(let e=0,n=this.count;e<n;e++)Ie.fromBufferAttribute(this,e),Ie.applyMatrix4(t),this.setXYZ(e,Ie.x,Ie.y,Ie.z);return this}applyNormalMatrix(t){for(let e=0,n=this.count;e<n;e++)Ie.fromBufferAttribute(this,e),Ie.applyNormalMatrix(t),this.setXYZ(e,Ie.x,Ie.y,Ie.z);return this}transformDirection(t){for(let e=0,n=this.count;e<n;e++)Ie.fromBufferAttribute(this,e),Ie.transformDirection(t),this.setXYZ(e,Ie.x,Ie.y,Ie.z);return this}set(t,e=0){return this.array.set(t,e),this}getComponent(t,e){let n=this.array[t*this.itemSize+e];return this.normalized&&(n=this._denormalize(n)),n}setComponent(t,e,n){return this.normalized&&(n=this._normalize(n)),this.array[t*this.itemSize+e]=n,this}getX(t){let e=this.array[t*this.itemSize];return this.normalized&&(e=this._denormalize(e)),e}setX(t,e){return this.normalized&&(e=this._normalize(e)),this.array[t*this.itemSize]=e,this}getY(t){let e=this.array[t*this.itemSize+1];return this.normalized&&(e=this._denormalize(e)),e}setY(t,e){return this.normalized&&(e=this._normalize(e)),this.array[t*this.itemSize+1]=e,this}getZ(t){let e=this.array[t*this.itemSize+2];return this.normalized&&(e=this._denormalize(e)),e}setZ(t,e){return this.normalized&&(e=this._normalize(e)),this.array[t*this.itemSize+2]=e,this}getW(t){let e=this.array[t*this.itemSize+3];return this.normalized&&(e=this._denormalize(e)),e}setW(t,e){return this.normalized&&(e=this._normalize(e)),this.array[t*this.itemSize+3]=e,this}setXY(t,e,n){return t*=this.itemSize,this.normalized&&(e=this._normalize(e),n=this._normalize(n)),this.array[t+0]=e,this.array[t+1]=n,this}setXYZ(t,e,n,s){return t*=this.itemSize,this.normalized&&(e=this._normalize(e),n=this._normalize(n),s=this._normalize(s)),this.array[t+0]=e,this.array[t+1]=n,this.array[t+2]=s,this}setXYZW(t,e,n,s,r){return t*=this.itemSize,this.normalized&&(e=this._normalize(e),n=this._normalize(n),s=this._normalize(s),r=this._normalize(r)),this.array[t+0]=e,this.array[t+1]=n,this.array[t+2]=s,this.array[t+3]=r,this}onUpload(t){return this.onUploadCallback=t,this}clone(){return new this.constructor(this.array,this.itemSize).copy(this)}toJSON(){let t={itemSize:this.itemSize,type:this.array.constructor.name,array:Array.from(this.array),normalized:this.normalized};return t.name=this.name,t.usage=this.usage,t.gpuType=this.gpuType,t}dispose(){this.dispatchEvent({type:"dispose"})}_denormalize(t){let e=this._normalizationScale;return e!==void 0?Math.max(t/e,-1):Iu(t,this.array)}_normalize(t){let e=this._normalizationScale;return e!==void 0?Math.round(t*e):Lu(t,this.array)}};var lr=class extends Mn{constructor(t,e,n){super(new Uint16Array(t),e,n)}};var cr=class extends Mn{constructor(t,e,n){super(new Uint32Array(t),e,n)}};var Jt=class extends Mn{constructor(t,e,n){super(new Float32Array(t),e,n)}},Nf=new Fn,Xs=new F,Tl=new F,Qn=class{constructor(t=new F,e=-1){this.isSphere=!0,this.center=t,this.radius=e}set(t,e){return this.center.copy(t),this.radius=e,this}setFromPoints(t,e){let n=this.center;e!==void 0?n.copy(e):Nf.setFromPoints(t).getCenter(n);let s=0;for(let r=0,a=t.length;r<a;r++)s=Math.max(s,n.distanceToSquared(t[r]));return this.radius=Math.sqrt(s),this}copy(t){return this.center.copy(t.center),this.radius=t.radius,this}isEmpty(){return this.radius<0}makeEmpty(){return this.center.set(0,0,0),this.radius=-1,this}containsPoint(t){return t.distanceToSquared(this.center)<=this.radius*this.radius}distanceToPoint(t){return t.distanceTo(this.center)-this.radius}intersectsSphere(t){let e=this.radius+t.radius;return t.center.distanceToSquared(this.center)<=e*e}intersectsBox(t){return t.intersectsSphere(this)}intersectsPlane(t){return Math.abs(t.distanceToPoint(this.center))<=this.radius}clampPoint(t,e){let n=this.center.distanceToSquared(t);return e.copy(t),n>this.radius*this.radius&&(e.sub(this.center).normalize(),e.multiplyScalar(this.radius).add(this.center)),e}getBoundingBox(t){return this.isEmpty()?(t.makeEmpty(),t):(t.set(this.center,this.center),t.expandByScalar(this.radius),t)}applyMatrix4(t){return this.center.applyMatrix4(t),this.radius=this.radius*t.getMaxScaleOnAxis(),this}translate(t){return this.center.add(t),this}expandByPoint(t){if(this.isEmpty())return this.center.copy(t),this.radius=0,this;Xs.subVectors(t,this.center);let e=Xs.lengthSq();if(e>this.radius*this.radius){let n=Math.sqrt(e),s=(n-this.radius)*.5;this.center.addScaledVector(Xs,s/n),this.radius+=s}return this}union(t){return t.isEmpty()?this:this.isEmpty()?(this.copy(t),this):(this.center.equals(t.center)===!0?this.radius=Math.max(this.radius,t.radius):(Tl.subVectors(t.center,this.center).setLength(t.radius),this.expandByPoint(Xs.copy(t.center).add(Tl)),this.expandByPoint(Xs.copy(t.center).sub(Tl))),this)}equals(t){return t.center.equals(this.center)&&t.radius===this.radius}clone(){return new this.constructor().copy(this)}toJSON(){return{radius:this.radius,center:this.center.toArray()}}fromJSON(t){return this.radius=t.radius,this.center.fromArray(t.center),this}},Uf=0,pn=new xe,Al=new Ne,ls=new F,an=new Fn,qs=new Fn,ze=new F,de=class i extends Sn{constructor(){super(),this.isBufferGeometry=!0,Object.defineProperty(this,"id",{value:Uf++}),this.uuid=Zi(),this.name="",this.type="BufferGeometry",this.index=null,this.indirect=null,this.indirectOffset=0,this.attributes={},this.morphAttributes={},this.morphTargetsRelative=!1,this.groups=[],this.boundingBox=null,this.boundingSphere=null,this.drawRange={start:0,count:1/0},this.userData={},this._transformed=!1}getIndex(){return this.index}setIndex(t){return Array.isArray(t)?this.index=new(af(t)?cr:lr)(t,1):this.index=t,this}setIndirect(t,e=0){return this.indirect=t,this.indirectOffset=e,this}getIndirect(){return this.indirect}getAttribute(t){return this.attributes[t]}setAttribute(t,e){return this.attributes[t]=e,this}deleteAttribute(t){return delete this.attributes[t],this}hasAttribute(t){return this.attributes[t]!==void 0}addGroup(t,e,n=0){this.groups.push({start:t,count:e,materialIndex:n})}clearGroups(){this.groups=[]}setDrawRange(t,e){this.drawRange.start=t,this.drawRange.count=e}applyMatrix4(t){let e=this.attributes.position;e!==void 0&&(e.applyMatrix4(t),e.needsUpdate=!0);let n=this.attributes.normal;if(n!==void 0){let r=new Xt().getNormalMatrix(t);n.applyNormalMatrix(r),n.needsUpdate=!0}let s=this.attributes.tangent;return s!==void 0&&(s.transformDirection(t),s.needsUpdate=!0),this.boundingBox!==null&&this.computeBoundingBox(),this.boundingSphere!==null&&this.computeBoundingSphere(),this._transformed=!0,this}applyQuaternion(t){return pn.makeRotationFromQuaternion(t),this.applyMatrix4(pn),this}rotateX(t){return pn.makeRotationX(t),this.applyMatrix4(pn),this}rotateY(t){return pn.makeRotationY(t),this.applyMatrix4(pn),this}rotateZ(t){return pn.makeRotationZ(t),this.applyMatrix4(pn),this}translate(t,e,n){return pn.makeTranslation(t,e,n),this.applyMatrix4(pn),this}scale(t,e,n){return pn.makeScale(t,e,n),this.applyMatrix4(pn),this}lookAt(t){return Al.lookAt(t),Al.updateMatrix(),this.applyMatrix4(Al.matrix),this}center(){return this.computeBoundingBox(),this.boundingBox.getCenter(ls).negate(),this.translate(ls.x,ls.y,ls.z),this}setFromPoints(t){let e=this.getAttribute("position");if(e===void 0){let n=[];for(let s=0,r=t.length;s<r;s++){let a=t[s];n.push(a.x,a.y,a.z||0)}this.setAttribute("position",new Jt(n,3))}else{let n=Math.min(t.length,e.count);for(let s=0;s<n;s++){let r=t[s];e.setXYZ(s,r.x,r.y,r.z||0)}t.length>e.count&&qt("BufferGeometry: Buffer size too small for points data. Use .dispose() and create a new geometry."),e.needsUpdate=!0}return this}computeBoundingBox(){this.boundingBox===null&&(this.boundingBox=new Fn);let t=this.attributes.position,e=this.morphAttributes.position;if(t&&t.isGLBufferAttribute){Zt("BufferGeometry.computeBoundingBox(): GLBufferAttribute requires a manual bounding box.",this),this.boundingBox.set(new F(-1/0,-1/0,-1/0),new F(1/0,1/0,1/0));return}if(t!==void 0){if(this.boundingBox.setFromBufferAttribute(t),e)for(let n=0,s=e.length;n<s;n++){let r=e[n];an.setFromBufferAttribute(r),this.morphTargetsRelative?(ze.addVectors(this.boundingBox.min,an.min),this.boundingBox.expandByPoint(ze),ze.addVectors(this.boundingBox.max,an.max),this.boundingBox.expandByPoint(ze)):(this.boundingBox.expandByPoint(an.min),this.boundingBox.expandByPoint(an.max))}}else this.boundingBox.makeEmpty();(isNaN(this.boundingBox.min.x)||isNaN(this.boundingBox.min.y)||isNaN(this.boundingBox.min.z))&&Zt('BufferGeometry.computeBoundingBox(): Computed min/max have NaN values. The "position" attribute is likely to have NaN values.',this)}computeBoundingSphere(){this.boundingSphere===null&&(this.boundingSphere=new Qn);let t=this.attributes.position,e=this.morphAttributes.position;if(t&&t.isGLBufferAttribute){Zt("BufferGeometry.computeBoundingSphere(): GLBufferAttribute requires a manual bounding sphere.",this),this.boundingSphere.set(new F,1/0);return}if(t){let n=this.boundingSphere.center;if(an.setFromBufferAttribute(t),e)for(let r=0,a=e.length;r<a;r++){let o=e[r];qs.setFromBufferAttribute(o),this.morphTargetsRelative?(ze.addVectors(an.min,qs.min),an.expandByPoint(ze),ze.addVectors(an.max,qs.max),an.expandByPoint(ze)):(an.expandByPoint(qs.min),an.expandByPoint(qs.max))}an.getCenter(n);let s=0;for(let r=0,a=t.count;r<a;r++)ze.fromBufferAttribute(t,r),s=Math.max(s,n.distanceToSquared(ze));if(e)for(let r=0,a=e.length;r<a;r++){let o=e[r],c=this.morphTargetsRelative;for(let l=0,u=o.count;l<u;l++)ze.fromBufferAttribute(o,l),c&&(ls.fromBufferAttribute(t,l),ze.add(ls)),s=Math.max(s,n.distanceToSquared(ze))}this.boundingSphere.radius=Math.sqrt(s),isNaN(this.boundingSphere.radius)&&Zt('BufferGeometry.computeBoundingSphere(): Computed radius is NaN. The "position" attribute is likely to have NaN values.',this)}}computeTangents(){let t=this.index,e=this.attributes;if(t===null||e.position===void 0||e.normal===void 0||e.uv===void 0){Zt("BufferGeometry: .computeTangents() failed. Missing required attributes (index, position, normal or uv)");return}let n=e.position,s=e.normal,r=e.uv,a=this.getAttribute("tangent");(a===void 0||a.count!==n.count)&&(a=new Mn(new Float32Array(4*n.count),4),this.setAttribute("tangent",a));let o=[],c=[],l=new Uint8Array(n.count);for(let P=0;P<n.count;P++)o[P]=new F,c[P]=new F;let u=new F,h=new F,d=new F,f=new ft,m=new ft,y=new ft,g=new F,p=new F;function v(P,C,k){l[P]=l[C]=l[k]=1,u.fromBufferAttribute(n,P),h.fromBufferAttribute(n,C),d.fromBufferAttribute(n,k),f.fromBufferAttribute(r,P),m.fromBufferAttribute(r,C),y.fromBufferAttribute(r,k),h.sub(u),d.sub(u),m.sub(f),y.sub(f);let O=1/(m.x*y.y-y.x*m.y);isFinite(O)&&(g.copy(h).multiplyScalar(y.y).addScaledVector(d,-m.y).multiplyScalar(O),p.copy(d).multiplyScalar(m.x).addScaledVector(h,-y.x).multiplyScalar(O),o[P].add(g),o[C].add(g),o[k].add(g),c[P].add(p),c[C].add(p),c[k].add(p))}let b=this.groups;b.length===0&&(b=[{start:0,count:t.count}]);for(let P=0,C=b.length;P<C;++P){let k=b[P],O=k.start,j=k.count;for(let D=O,K=O+j;D<K;D+=3)v(t.getX(D+0),t.getX(D+1),t.getX(D+2))}let _=new F,T=new F,w=new F,x=new F;function M(P){w.fromBufferAttribute(s,P),x.copy(w);let C=o[P];_.copy(C),_.sub(w.multiplyScalar(w.dot(C))).normalize(),T.crossVectors(x,C);let O=T.dot(c[P])<0?-1:1;a.setXYZW(P,_.x,_.y,_.z,O)}for(let P=0,C=n.count;P<C;++P)l[P]&&M(P);this._transformed=!0}computeVertexNormals(){let t=this.index,e=this.getAttribute("position");if(e!==void 0){let n=this.getAttribute("normal");if(n===void 0||n.count!==e.count)n=new Mn(new Float32Array(e.count*3),3),this.setAttribute("normal",n);else for(let d=0,f=n.count;d<f;d++)n.setXYZ(d,0,0,0);let s=new F,r=new F,a=new F,o=new F,c=new F,l=new F,u=new F,h=new F;if(t)for(let d=0,f=t.count;d<f;d+=3){let m=t.getX(d+0),y=t.getX(d+1),g=t.getX(d+2);s.fromBufferAttribute(e,m),r.fromBufferAttribute(e,y),a.fromBufferAttribute(e,g),u.subVectors(a,r),h.subVectors(s,r),u.cross(h),o.fromBufferAttribute(n,m),c.fromBufferAttribute(n,y),l.fromBufferAttribute(n,g),o.add(u),c.add(u),l.add(u),n.setXYZ(m,o.x,o.y,o.z),n.setXYZ(y,c.x,c.y,c.z),n.setXYZ(g,l.x,l.y,l.z)}else for(let d=0,f=e.count;d<f;d+=3)s.fromBufferAttribute(e,d+0),r.fromBufferAttribute(e,d+1),a.fromBufferAttribute(e,d+2),u.subVectors(a,r),h.subVectors(s,r),u.cross(h),n.setXYZ(d+0,u.x,u.y,u.z),n.setXYZ(d+1,u.x,u.y,u.z),n.setXYZ(d+2,u.x,u.y,u.z);this.normalizeNormals(),n.needsUpdate=!0}}normalizeNormals(){let t=this.attributes.normal;for(let e=0,n=t.count;e<n;e++)ze.fromBufferAttribute(t,e),ze.normalize(),t.setXYZ(e,ze.x,ze.y,ze.z)}toNonIndexed(){function t(o,c){let l=o.array,u=o.itemSize,h=o.normalized,d=new l.constructor(c.length*u),f=0,m=0;for(let y=0,g=c.length;y<g;y++){o.isInterleavedBufferAttribute?f=c[y]*o.data.stride+o.offset:f=c[y]*u;for(let p=0;p<u;p++)d[m++]=l[f++]}return new Mn(d,u,h)}if(this.index===null)return qt("BufferGeometry.toNonIndexed(): BufferGeometry is already non-indexed."),this;let e=new i,n=this.index.array,s=this.attributes;for(let o in s){let c=s[o],l=t(c,n);e.setAttribute(o,l)}let r=this.morphAttributes;for(let o in r){let c=[],l=r[o];for(let u=0,h=l.length;u<h;u++){let d=l[u],f=t(d,n);c.push(f)}e.morphAttributes[o]=c}e.morphTargetsRelative=this.morphTargetsRelative;let a=this.groups;for(let o=0,c=a.length;o<c;o++){let l=a[o];e.addGroup(l.start,l.count,l.materialIndex)}return e}toJSON(){let t={metadata:{version:4.7,type:"BufferGeometry",generator:"BufferGeometry.toJSON"}};if(t.uuid=this.uuid,t.type=this.parameters!==void 0&&this._transformed===!0?"BufferGeometry":this.type,t.name=this.name,Object.keys(this.userData).length>0&&(t.userData=this.userData),this.parameters!==void 0&&this._transformed!==!0){let c=this.parameters;for(let l in c)c[l]!==void 0&&(t[l]=c[l]);return t}t.data={attributes:{}};let e=this.index;e!==null&&(t.data.index={type:e.array.constructor.name,array:Array.prototype.slice.call(e.array)});let n=this.attributes;for(let c in n){let l=n[c];t.data.attributes[c]=l.toJSON(t.data)}let s={},r=!1;for(let c in this.morphAttributes){let l=this.morphAttributes[c],u=[];for(let h=0,d=l.length;h<d;h++){let f=l[h];u.push(f.toJSON(t.data))}u.length>0&&(s[c]=u,r=!0)}r&&(t.data.morphAttributes=s,t.data.morphTargetsRelative=this.morphTargetsRelative);let a=this.groups;a.length>0&&(t.data.groups=JSON.parse(JSON.stringify(a)));let o=this.boundingSphere;return o!==null&&(t.data.boundingSphere=o.toJSON()),t}clone(){return new this.constructor().copy(this)}copy(t){this.index=null,this.attributes={},this.morphAttributes={},this.groups=[],this.boundingBox=null,this.boundingSphere=null;let e={};this.name=t.name;let n=t.index;n!==null&&this.setIndex(n.clone());let s=t.attributes;for(let l in s){let u=s[l];this.setAttribute(l,u.clone(e))}let r=t.morphAttributes;for(let l in r){let u=[],h=r[l];for(let d=0,f=h.length;d<f;d++)u.push(h[d].clone(e));this.morphAttributes[l]=u}this.morphTargetsRelative=t.morphTargetsRelative;let a=t.groups;for(let l=0,u=a.length;l<u;l++){let h=a[l];this.addGroup(h.start,h.count,h.materialIndex)}let o=t.boundingBox;o!==null&&(this.boundingBox=o.clone());let c=t.boundingSphere;return c!==null&&(this.boundingSphere=c.clone()),this.drawRange.start=t.drawRange.start,this.drawRange.count=t.drawRange.count,this.userData=t.userData,this._transformed=t._transformed,this}dispose(){this.dispatchEvent({type:"dispose"})}};var El=new F,Ff=new F,Of=new Xt,on=class{constructor(t=new F(1,0,0),e=0){this.isPlane=!0,this.normal=t,this.constant=e}set(t,e){return this.normal.copy(t),this.constant=e,this}setComponents(t,e,n,s){return this.normal.set(t,e,n),this.constant=s,this}setFromNormalAndCoplanarPoint(t,e){return this.normal.copy(t),this.constant=-e.dot(this.normal),this}setFromCoplanarPoints(t,e,n){let s=El.subVectors(n,e).cross(Ff.subVectors(t,e)).normalize();return this.setFromNormalAndCoplanarPoint(s,t),this}copy(t){return this.normal.copy(t.normal),this.constant=t.constant,this}normalize(){let t=1/this.normal.length();return this.normal.multiplyScalar(t),this.constant*=t,this}negate(){return this.constant*=-1,this.normal.negate(),this}distanceToPoint(t){return this.normal.dot(t)+this.constant}distanceToSphere(t){return this.distanceToPoint(t.center)-t.radius}projectPoint(t,e){return e.copy(t).addScaledVector(this.normal,-this.distanceToPoint(t))}intersectLine(t,e,n=!0){let s=t.delta(El),r=this.normal.dot(s);if(r===0)return this.distanceToPoint(t.start)===0?e.copy(t.start):null;let a=-(t.start.dot(this.normal)+this.constant)/r;return n===!0&&(a<0||a>1)?null:e.copy(t.start).addScaledVector(s,a)}intersectsLine(t){let e=this.distanceToPoint(t.start),n=this.distanceToPoint(t.end);return e<0&&n>0||n<0&&e>0}intersectsBox(t){return t.intersectsPlane(this)}intersectsSphere(t){return t.intersectsPlane(this)}coplanarPoint(t){return t.copy(this.normal).multiplyScalar(-this.constant)}applyMatrix4(t,e){let n=e||Of.getNormalMatrix(t),s=this.coplanarPoint(El).applyMatrix4(t),r=this.normal.applyMatrix3(n).normalize();return this.constant=-s.dot(r),this}translate(t){return this.constant-=t.dot(this.normal),this}equals(t){return t.normal.equals(this.normal)&&t.constant===this.constant}clone(){return new this.constructor().copy(this)}toJSON(){return{normal:this.normal.toArray(),constant:this.constant}}fromJSON(t){return this.normal.fromArray(t.normal),this.constant=t.constant,this}},Bf=0,ti=class extends Sn{constructor(){super(),this.isMaterial=!0,Object.defineProperty(this,"id",{value:Bf++}),this.uuid=Zi(),this.name="",this.type="Material",this.blending=Ds,this.side=wi,this.vertexColors=!1,this.opacity=1,this.transparent=!1,this.alphaHash=!1,this.blendSrc=Ql,this.blendDst=tc,this.blendEquation=qi,this.blendSrcAlpha=null,this.blendDstAlpha=null,this.blendEquationAlpha=null,this.blendColor=new Vt(0,0,0),this.blendAlpha=0,this.depthFunc=ps,this.depthTest=!0,this.depthWrite=!0,this.stencilWriteMask=255,this.stencilFunc=vu,this.stencilRef=0,this.stencilFuncMask=255,this.stencilFail=Ca,this.stencilZFail=Ca,this.stencilZPass=Ca,this.stencilWrite=!1,this.clippingPlanes=null,this.clipIntersection=!1,this.clipShadows=!1,this.shadowSide=null,this.colorWrite=!0,this.precision=null,this.polygonOffset=!1,this.polygonOffsetFactor=0,this.polygonOffsetUnits=0,this.dithering=!1,this.alphaToCoverage=!1,this.premultipliedAlpha=!1,this.forceSinglePass=!1,this.allowOverride=!0,this.visible=!0,this.toneMapped=!0,this.userData={},this.version=0,this._alphaTest=0}get alphaTest(){return this._alphaTest}set alphaTest(t){this._alphaTest>0!=t>0&&this.version++,this._alphaTest=t}onBeforeRender(){}onBeforeCompile(){}customProgramCacheKey(){return this.onBeforeCompile.toString()}setValues(t){if(t!==void 0)for(let e in t){let n=t[e];if(n===void 0){qt(`Material: parameter '${e}' has value of undefined.`);continue}let s=this[e];if(s===void 0){qt(`Material: '${e}' is not a property of THREE.${this.type}.`);continue}s&&s.isColor?s.set(n):s&&s.isVector2&&n&&n.isVector2||s&&s.isEuler&&n&&n.isEuler||s&&s.isVector3&&n&&n.isVector3?s.copy(n):this[e]=n}}toJSON(t){let e=t===void 0||typeof t=="string";e&&(t={textures:{},images:{}});let n={metadata:{version:4.7,type:"Material",generator:"Material.toJSON"}};n.uuid=this.uuid,n.type=this.type,n.blending=this.blending,n.side=this.side,n.shadowSide=this.shadowSide,n.vertexColors=this.vertexColors,n.opacity=this.opacity,n.transparent=this.transparent,n.blendSrc=this.blendSrc,n.blendDst=this.blendDst,n.blendEquation=this.blendEquation,n.blendSrcAlpha=this.blendSrcAlpha,n.blendDstAlpha=this.blendDstAlpha,n.blendEquationAlpha=this.blendEquationAlpha,n.blendColor=this.blendColor.getHex(),n.blendAlpha=this.blendAlpha,n.depthFunc=this.depthFunc,n.depthTest=this.depthTest,n.depthWrite=this.depthWrite,n.colorWrite=this.colorWrite,n.clipIntersection=this.clipIntersection,n.clipShadows=this.clipShadows,n.stencilWriteMask=this.stencilWriteMask,n.stencilFunc=this.stencilFunc,n.stencilRef=this.stencilRef,n.stencilFuncMask=this.stencilFuncMask,n.stencilFail=this.stencilFail,n.stencilZFail=this.stencilZFail,n.stencilZPass=this.stencilZPass,n.stencilWrite=this.stencilWrite,n.polygonOffset=this.polygonOffset,n.polygonOffsetFactor=this.polygonOffsetFactor,n.polygonOffsetUnits=this.polygonOffsetUnits,n.dithering=this.dithering,n.alphaTest=this.alphaTest,n.alphaHash=this.alphaHash,n.alphaToCoverage=this.alphaToCoverage,n.premultipliedAlpha=this.premultipliedAlpha,n.forceSinglePass=this.forceSinglePass,n.allowOverride=this.allowOverride,n.visible=this.visible,n.toneMapped=this.toneMapped,n.name=this.name,this.color&&this.color.isColor&&(n.color=this.color.getHex()),this.roughness!==void 0&&(n.roughness=this.roughness),this.metalness!==void 0&&(n.metalness=this.metalness),this.sheen!==void 0&&(n.sheen=this.sheen),this.sheenColor&&this.sheenColor.isColor&&(n.sheenColor=this.sheenColor.getHex()),this.sheenRoughness!==void 0&&(n.sheenRoughness=this.sheenRoughness),this.emissive&&this.emissive.isColor&&(n.emissive=this.emissive.getHex()),this.emissiveIntensity!==void 0&&(n.emissiveIntensity=this.emissiveIntensity),this.specular&&this.specular.isColor&&(n.specular=this.specular.getHex()),this.specularIntensity!==void 0&&(n.specularIntensity=this.specularIntensity),this.specularColor&&this.specularColor.isColor&&(n.specularColor=this.specularColor.getHex()),this.shininess!==void 0&&(n.shininess=this.shininess),this.clearcoat!==void 0&&(n.clearcoat=this.clearcoat),this.clearcoatRoughness!==void 0&&(n.clearcoatRoughness=this.clearcoatRoughness),this.diffuseRoughness!==void 0&&(n.diffuseRoughness=this.diffuseRoughness),this.diffuseRoughnessMap&&this.diffuseRoughnessMap.isTexture&&(n.diffuseRoughnessMap=this.diffuseRoughnessMap.toJSON(t).uuid),this.clearcoatMap&&this.clearcoatMap.isTexture&&(n.clearcoatMap=this.clearcoatMap.toJSON(t).uuid),this.clearcoatRoughnessMap&&this.clearcoatRoughnessMap.isTexture&&(n.clearcoatRoughnessMap=this.clearcoatRoughnessMap.toJSON(t).uuid),this.clearcoatNormalMap&&this.clearcoatNormalMap.isTexture&&(n.clearcoatNormalMap=this.clearcoatNormalMap.toJSON(t).uuid,n.clearcoatNormalScale=this.clearcoatNormalScale.toArray()),this.sheenColorMap&&this.sheenColorMap.isTexture&&(n.sheenColorMap=this.sheenColorMap.toJSON(t).uuid),this.sheenRoughnessMap&&this.sheenRoughnessMap.isTexture&&(n.sheenRoughnessMap=this.sheenRoughnessMap.toJSON(t).uuid),this.dispersion!==void 0&&(n.dispersion=this.dispersion),this.retroreflectivity!==void 0&&(n.retroreflectivity=this.retroreflectivity),this.iridescence!==void 0&&(n.iridescence=this.iridescence),this.iridescenceIOR!==void 0&&(n.iridescenceIOR=this.iridescenceIOR),this.iridescenceThicknessMinimum!==void 0&&(n.iridescenceThicknessMinimum=this.iridescenceThicknessMinimum),this.iridescenceThicknessMaximum!==void 0&&(n.iridescenceThicknessMaximum=this.iridescenceThicknessMaximum),this.iridescenceMap&&this.iridescenceMap.isTexture&&(n.iridescenceMap=this.iridescenceMap.toJSON(t).uuid),this.iridescenceThicknessMap&&this.iridescenceThicknessMap.isTexture&&(n.iridescenceThicknessMap=this.iridescenceThicknessMap.toJSON(t).uuid),this.anisotropy!==void 0&&(n.anisotropy=this.anisotropy),this.anisotropyRotation!==void 0&&(n.anisotropyRotation=this.anisotropyRotation),this.anisotropyMap&&this.anisotropyMap.isTexture&&(n.anisotropyMap=this.anisotropyMap.toJSON(t).uuid),this.map&&this.map.isTexture&&(n.map=this.map.toJSON(t).uuid),this.matcap&&this.matcap.isTexture&&(n.matcap=this.matcap.toJSON(t).uuid),this.alphaMap&&this.alphaMap.isTexture&&(n.alphaMap=this.alphaMap.toJSON(t).uuid),this.lightMap&&this.lightMap.isTexture&&(n.lightMap=this.lightMap.toJSON(t).uuid,n.lightMapIntensity=this.lightMapIntensity),this.aoMap&&this.aoMap.isTexture&&(n.aoMap=this.aoMap.toJSON(t).uuid,n.aoMapIntensity=this.aoMapIntensity),this.bumpMap&&this.bumpMap.isTexture&&(n.bumpMap=this.bumpMap.toJSON(t).uuid,n.bumpScale=this.bumpScale),this.normalMap&&this.normalMap.isTexture&&(n.normalMap=this.normalMap.toJSON(t).uuid,n.normalMapType=this.normalMapType,n.normalScale=this.normalScale.toArray()),this.displacementMap&&this.displacementMap.isTexture&&(n.displacementMap=this.displacementMap.toJSON(t).uuid,n.displacementScale=this.displacementScale,n.displacementBias=this.displacementBias),this.roughnessMap&&this.roughnessMap.isTexture&&(n.roughnessMap=this.roughnessMap.toJSON(t).uuid),this.metalnessMap&&this.metalnessMap.isTexture&&(n.metalnessMap=this.metalnessMap.toJSON(t).uuid),this.emissiveMap&&this.emissiveMap.isTexture&&(n.emissiveMap=this.emissiveMap.toJSON(t).uuid),this.specularMap&&this.specularMap.isTexture&&(n.specularMap=this.specularMap.toJSON(t).uuid),this.specularIntensityMap&&this.specularIntensityMap.isTexture&&(n.specularIntensityMap=this.specularIntensityMap.toJSON(t).uuid),this.specularColorMap&&this.specularColorMap.isTexture&&(n.specularColorMap=this.specularColorMap.toJSON(t).uuid),this.envMap&&this.envMap.isTexture&&(n.envMap=this.envMap.toJSON(t).uuid,this.combine!==void 0&&(n.combine=this.combine)),this.envMapRotation!==void 0&&(n.envMapRotation=this.envMapRotation.toArray()),this.envMapIntensity!==void 0&&(n.envMapIntensity=this.envMapIntensity),this.reflectivity!==void 0&&(n.reflectivity=this.reflectivity),this.refractionRatio!==void 0&&(n.refractionRatio=this.refractionRatio),this.gradientMap&&this.gradientMap.isTexture&&(n.gradientMap=this.gradientMap.toJSON(t).uuid),this.transmission!==void 0&&(n.transmission=this.transmission),this.transmissionMap&&this.transmissionMap.isTexture&&(n.transmissionMap=this.transmissionMap.toJSON(t).uuid),this.thickness!==void 0&&(n.thickness=this.thickness),this.thicknessMap&&this.thicknessMap.isTexture&&(n.thicknessMap=this.thicknessMap.toJSON(t).uuid),this.attenuationDistance!==void 0&&(n.attenuationDistance=this.attenuationDistance),this.attenuationColor!==void 0&&(n.attenuationColor=this.attenuationColor.getHex()),this.size!==void 0&&(n.size=this.size),this.sizeAttenuation!==void 0&&(n.sizeAttenuation=this.sizeAttenuation),Array.isArray(this.clippingPlanes)&&this.clippingPlanes.length>0&&(n.clippingPlanes=this.clippingPlanes.map(r=>r.toJSON())),this.rotation!==void 0&&(n.rotation=this.rotation),this.depthPacking!==void 0&&(n.depthPacking=this.depthPacking),this.linewidth!==void 0&&(n.linewidth=this.linewidth),this.linecap!==void 0&&(n.linecap=this.linecap),this.linejoin!==void 0&&(n.linejoin=this.linejoin),this.dashSize!==void 0&&(n.dashSize=this.dashSize),this.gapSize!==void 0&&(n.gapSize=this.gapSize),this.scale!==void 0&&(n.scale=this.scale),this.wireframe!==void 0&&(n.wireframe=this.wireframe),this.wireframeLinewidth!==void 0&&(n.wireframeLinewidth=this.wireframeLinewidth),this.wireframeLinecap!==void 0&&(n.wireframeLinecap=this.wireframeLinecap),this.wireframeLinejoin!==void 0&&(n.wireframeLinejoin=this.wireframeLinejoin),this.flatShading!==void 0&&(n.flatShading=this.flatShading),this.fog!==void 0&&(n.fog=this.fog),Object.keys(this.userData).length>0&&(n.userData=this.userData);function s(r){let a=[];for(let o in r){let c=r[o];delete c.metadata,a.push(c)}return a}if(e){let r=s(t.textures),a=s(t.images);r.length>0&&(n.textures=r),a.length>0&&(n.images=a)}return n}fromJSON(t,e){if(t.uuid!==void 0&&(this.uuid=t.uuid),t.name!==void 0&&(this.name=t.name),t.color!==void 0&&this.color!==void 0&&this.color.setHex(t.color),t.roughness!==void 0&&(this.roughness=t.roughness),t.metalness!==void 0&&(this.metalness=t.metalness),t.sheen!==void 0&&(this.sheen=t.sheen),t.sheenColor!==void 0&&(this.sheenColor=new Vt().setHex(t.sheenColor)),t.sheenRoughness!==void 0&&(this.sheenRoughness=t.sheenRoughness),t.emissive!==void 0&&this.emissive!==void 0&&this.emissive.setHex(t.emissive),t.specular!==void 0&&this.specular!==void 0&&this.specular.setHex(t.specular),t.specularIntensity!==void 0&&(this.specularIntensity=t.specularIntensity),t.specularColor!==void 0&&this.specularColor!==void 0&&this.specularColor.setHex(t.specularColor),t.shininess!==void 0&&(this.shininess=t.shininess),t.clearcoat!==void 0&&(this.clearcoat=t.clearcoat),t.clearcoatRoughness!==void 0&&(this.clearcoatRoughness=t.clearcoatRoughness),t.diffuseRoughness!==void 0&&(this.diffuseRoughness=t.diffuseRoughness),t.dispersion!==void 0&&(this.dispersion=t.dispersion),t.retroreflectivity!==void 0&&(this.retroreflectivity=t.retroreflectivity),t.iridescence!==void 0&&(this.iridescence=t.iridescence),t.iridescenceIOR!==void 0&&(this.iridescenceIOR=t.iridescenceIOR),t.iridescenceThicknessMinimum!==void 0&&(this.iridescenceThicknessMinimum=t.iridescenceThicknessMinimum),t.iridescenceThicknessMaximum!==void 0&&(this.iridescenceThicknessMaximum=t.iridescenceThicknessMaximum),t.iridescenceThicknessRange!==void 0&&(this.iridescenceThicknessMinimum=t.iridescenceThicknessRange[0],this.iridescenceThicknessMaximum=t.iridescenceThicknessRange[1]),t.transmission!==void 0&&(this.transmission=t.transmission),t.thickness!==void 0&&(this.thickness=t.thickness),t.attenuationDistance!==void 0&&(this.attenuationDistance=t.attenuationDistance),t.attenuationColor!==void 0&&this.attenuationColor!==void 0&&this.attenuationColor.setHex(t.attenuationColor),t.anisotropy!==void 0&&(this.anisotropy=t.anisotropy),t.anisotropyRotation!==void 0&&(this.anisotropyRotation=t.anisotropyRotation),t.fog!==void 0&&(this.fog=t.fog),t.flatShading!==void 0&&(this.flatShading=t.flatShading),t.blending!==void 0&&(this.blending=t.blending),t.combine!==void 0&&(this.combine=t.combine),t.side!==void 0&&(this.side=t.side),t.shadowSide!==void 0&&(this.shadowSide=t.shadowSide),t.opacity!==void 0&&(this.opacity=t.opacity),t.transparent!==void 0&&(this.transparent=t.transparent),t.alphaTest!==void 0&&(this.alphaTest=t.alphaTest),t.alphaHash!==void 0&&(this.alphaHash=t.alphaHash),t.depthFunc!==void 0&&(this.depthFunc=t.depthFunc),t.depthTest!==void 0&&(this.depthTest=t.depthTest),t.depthWrite!==void 0&&(this.depthWrite=t.depthWrite),t.colorWrite!==void 0&&(this.colorWrite=t.colorWrite),t.clippingPlanes!==void 0&&(this.clippingPlanes=t.clippingPlanes.map(n=>new on().fromJSON(n))),t.clipIntersection!==void 0&&(this.clipIntersection=t.clipIntersection),t.clipShadows!==void 0&&(this.clipShadows=t.clipShadows),t.depthPacking!==void 0&&(this.depthPacking=t.depthPacking),t.blendSrc!==void 0&&(this.blendSrc=t.blendSrc),t.blendDst!==void 0&&(this.blendDst=t.blendDst),t.blendEquation!==void 0&&(this.blendEquation=t.blendEquation),t.blendSrcAlpha!==void 0&&(this.blendSrcAlpha=t.blendSrcAlpha),t.blendDstAlpha!==void 0&&(this.blendDstAlpha=t.blendDstAlpha),t.blendEquationAlpha!==void 0&&(this.blendEquationAlpha=t.blendEquationAlpha),t.blendColor!==void 0&&this.blendColor!==void 0&&this.blendColor.setHex(t.blendColor),t.blendAlpha!==void 0&&(this.blendAlpha=t.blendAlpha),t.stencilWriteMask!==void 0&&(this.stencilWriteMask=t.stencilWriteMask),t.stencilFunc!==void 0&&(this.stencilFunc=t.stencilFunc),t.stencilRef!==void 0&&(this.stencilRef=t.stencilRef),t.stencilFuncMask!==void 0&&(this.stencilFuncMask=t.stencilFuncMask),t.stencilFail!==void 0&&(this.stencilFail=t.stencilFail),t.stencilZFail!==void 0&&(this.stencilZFail=t.stencilZFail),t.stencilZPass!==void 0&&(this.stencilZPass=t.stencilZPass),t.stencilWrite!==void 0&&(this.stencilWrite=t.stencilWrite),t.wireframe!==void 0&&(this.wireframe=t.wireframe),t.wireframeLinewidth!==void 0&&(this.wireframeLinewidth=t.wireframeLinewidth),t.wireframeLinecap!==void 0&&(this.wireframeLinecap=t.wireframeLinecap),t.wireframeLinejoin!==void 0&&(this.wireframeLinejoin=t.wireframeLinejoin),t.rotation!==void 0&&(this.rotation=t.rotation),t.linewidth!==void 0&&(this.linewidth=t.linewidth),t.linecap!==void 0&&(this.linecap=t.linecap),t.linejoin!==void 0&&(this.linejoin=t.linejoin),t.dashSize!==void 0&&(this.dashSize=t.dashSize),t.gapSize!==void 0&&(this.gapSize=t.gapSize),t.scale!==void 0&&(this.scale=t.scale),t.polygonOffset!==void 0&&(this.polygonOffset=t.polygonOffset),t.polygonOffsetFactor!==void 0&&(this.polygonOffsetFactor=t.polygonOffsetFactor),t.polygonOffsetUnits!==void 0&&(this.polygonOffsetUnits=t.polygonOffsetUnits),t.dithering!==void 0&&(this.dithering=t.dithering),t.alphaToCoverage!==void 0&&(this.alphaToCoverage=t.alphaToCoverage),t.premultipliedAlpha!==void 0&&(this.premultipliedAlpha=t.premultipliedAlpha),t.forceSinglePass!==void 0&&(this.forceSinglePass=t.forceSinglePass),t.allowOverride!==void 0&&(this.allowOverride=t.allowOverride),t.visible!==void 0&&(this.visible=t.visible),t.toneMapped!==void 0&&(this.toneMapped=t.toneMapped),t.userData!==void 0&&(this.userData=t.userData),t.vertexColors!==void 0&&(typeof t.vertexColors=="number"?this.vertexColors=t.vertexColors>0:this.vertexColors=t.vertexColors),t.size!==void 0&&(this.size=t.size),t.sizeAttenuation!==void 0&&(this.sizeAttenuation=t.sizeAttenuation),t.map!==void 0&&(this.map=e[t.map]||null),t.matcap!==void 0&&(this.matcap=e[t.matcap]||null),t.alphaMap!==void 0&&(this.alphaMap=e[t.alphaMap]||null),t.bumpMap!==void 0&&(this.bumpMap=e[t.bumpMap]||null),t.bumpScale!==void 0&&(this.bumpScale=t.bumpScale),t.normalMap!==void 0&&(this.normalMap=e[t.normalMap]||null),t.normalMapType!==void 0&&(this.normalMapType=t.normalMapType),t.normalScale!==void 0){let n=t.normalScale;Array.isArray(n)===!1&&(n=[n,n]),this.normalScale=new ft().fromArray(n)}return t.displacementMap!==void 0&&(this.displacementMap=e[t.displacementMap]||null),t.displacementScale!==void 0&&(this.displacementScale=t.displacementScale),t.displacementBias!==void 0&&(this.displacementBias=t.displacementBias),t.roughnessMap!==void 0&&(this.roughnessMap=e[t.roughnessMap]||null),t.metalnessMap!==void 0&&(this.metalnessMap=e[t.metalnessMap]||null),t.emissiveMap!==void 0&&(this.emissiveMap=e[t.emissiveMap]||null),t.emissiveIntensity!==void 0&&(this.emissiveIntensity=t.emissiveIntensity),t.specularMap!==void 0&&(this.specularMap=e[t.specularMap]||null),t.specularIntensityMap!==void 0&&(this.specularIntensityMap=e[t.specularIntensityMap]||null),t.specularColorMap!==void 0&&(this.specularColorMap=e[t.specularColorMap]||null),t.envMap!==void 0&&(this.envMap=e[t.envMap]||null),t.envMapRotation!==void 0&&this.envMapRotation.fromArray(t.envMapRotation),t.envMapIntensity!==void 0&&(this.envMapIntensity=t.envMapIntensity),t.reflectivity!==void 0&&(this.reflectivity=t.reflectivity),t.refractionRatio!==void 0&&(this.refractionRatio=t.refractionRatio),t.lightMap!==void 0&&(this.lightMap=e[t.lightMap]||null),t.lightMapIntensity!==void 0&&(this.lightMapIntensity=t.lightMapIntensity),t.aoMap!==void 0&&(this.aoMap=e[t.aoMap]||null),t.aoMapIntensity!==void 0&&(this.aoMapIntensity=t.aoMapIntensity),t.gradientMap!==void 0&&(this.gradientMap=e[t.gradientMap]||null),t.clearcoatMap!==void 0&&(this.clearcoatMap=e[t.clearcoatMap]||null),t.clearcoatRoughnessMap!==void 0&&(this.clearcoatRoughnessMap=e[t.clearcoatRoughnessMap]||null),t.clearcoatNormalMap!==void 0&&(this.clearcoatNormalMap=e[t.clearcoatNormalMap]||null),t.clearcoatNormalScale!==void 0&&(this.clearcoatNormalScale=new ft().fromArray(t.clearcoatNormalScale)),t.diffuseRoughnessMap!==void 0&&(this.diffuseRoughnessMap=e[t.diffuseRoughnessMap]||null),t.iridescenceMap!==void 0&&(this.iridescenceMap=e[t.iridescenceMap]||null),t.iridescenceThicknessMap!==void 0&&(this.iridescenceThicknessMap=e[t.iridescenceThicknessMap]||null),t.transmissionMap!==void 0&&(this.transmissionMap=e[t.transmissionMap]||null),t.thicknessMap!==void 0&&(this.thicknessMap=e[t.thicknessMap]||null),t.anisotropyMap!==void 0&&(this.anisotropyMap=e[t.anisotropyMap]||null),t.sheenColorMap!==void 0&&(this.sheenColorMap=e[t.sheenColorMap]||null),t.sheenRoughnessMap!==void 0&&(this.sheenRoughnessMap=e[t.sheenRoughnessMap]||null),this}clone(){return new this.constructor().copy(this)}copy(t){this.name=t.name,this.blending=t.blending,this.side=t.side,this.vertexColors=t.vertexColors,this.opacity=t.opacity,this.transparent=t.transparent,this.blendSrc=t.blendSrc,this.blendDst=t.blendDst,this.blendEquation=t.blendEquation,this.blendSrcAlpha=t.blendSrcAlpha,this.blendDstAlpha=t.blendDstAlpha,this.blendEquationAlpha=t.blendEquationAlpha,this.blendColor.copy(t.blendColor),this.blendAlpha=t.blendAlpha,this.depthFunc=t.depthFunc,this.depthTest=t.depthTest,this.depthWrite=t.depthWrite,this.stencilWriteMask=t.stencilWriteMask,this.stencilFunc=t.stencilFunc,this.stencilRef=t.stencilRef,this.stencilFuncMask=t.stencilFuncMask,this.stencilFail=t.stencilFail,this.stencilZFail=t.stencilZFail,this.stencilZPass=t.stencilZPass,this.stencilWrite=t.stencilWrite;let e=t.clippingPlanes,n=null;if(e!==null){let s=e.length;n=new Array(s);for(let r=0;r!==s;++r)n[r]=e[r].clone()}return this.clippingPlanes=n,this.clipIntersection=t.clipIntersection,this.clipShadows=t.clipShadows,this.shadowSide=t.shadowSide,this.colorWrite=t.colorWrite,this.precision=t.precision,this.polygonOffset=t.polygonOffset,this.polygonOffsetFactor=t.polygonOffsetFactor,this.polygonOffsetUnits=t.polygonOffsetUnits,this.dithering=t.dithering,this.alphaTest=t.alphaTest,this.alphaHash=t.alphaHash,this.alphaToCoverage=t.alphaToCoverage,this.premultipliedAlpha=t.premultipliedAlpha,this.forceSinglePass=t.forceSinglePass,this.allowOverride=t.allowOverride,this.visible=t.visible,this.toneMapped=t.toneMapped,this.userData=JSON.parse(JSON.stringify(t.userData)),this}dispose(){this.dispatchEvent({type:"dispose"})}set needsUpdate(t){t===!0&&this.version++}};var $n=new F,Cl=new F,ca=new F,ha=new F,xi=class{constructor(t=new F,e=new F(0,0,-1)){this.origin=t,this.direction=e}set(t,e){return this.origin.copy(t),this.direction.copy(e),this}copy(t){return this.origin.copy(t.origin),this.direction.copy(t.direction),this}at(t,e){return e.copy(this.origin).addScaledVector(this.direction,t)}lookAt(t){return this.direction.copy(t).sub(this.origin).normalize(),this}recast(t){return this.origin.copy(this.at(t,$n)),this}closestPointToPoint(t,e){e.subVectors(t,this.origin);let n=e.dot(this.direction);return n<0?e.copy(this.origin):e.copy(this.origin).addScaledVector(this.direction,n)}distanceToPoint(t){return Math.sqrt(this.distanceSqToPoint(t))}distanceSqToPoint(t){let e=$n.subVectors(t,this.origin).dot(this.direction);return e<0?this.origin.distanceToSquared(t):($n.copy(this.origin).addScaledVector(this.direction,e),$n.distanceToSquared(t))}distanceSqToSegment(t,e,n,s){Cl.copy(t).add(e).multiplyScalar(.5),ca.copy(e).sub(t).normalize(),ha.copy(this.origin).sub(Cl);let r=t.distanceTo(e)*.5,a=-this.direction.dot(ca),o=ha.dot(this.direction),c=-ha.dot(ca),l=ha.lengthSq(),u=Math.abs(1-a*a),h,d,f,m;if(u>0)if(h=a*c-o,d=a*o-c,m=r*u,h>=0)if(d>=-m)if(d<=m){let y=1/u;h*=y,d*=y,f=h*(h+a*d+2*o)+d*(a*h+d+2*c)+l}else d=r,h=Math.max(0,-(a*d+o)),f=-h*h+d*(d+2*c)+l;else d=-r,h=Math.max(0,-(a*d+o)),f=-h*h+d*(d+2*c)+l;else d<=-m?(h=Math.max(0,-(-a*r+o)),d=h>0?-r:Math.min(Math.max(-r,-c),r),f=-h*h+d*(d+2*c)+l):d<=m?(h=0,d=Math.min(Math.max(-r,-c),r),f=d*(d+2*c)+l):(h=Math.max(0,-(a*r+o)),d=h>0?r:Math.min(Math.max(-r,-c),r),f=-h*h+d*(d+2*c)+l);else d=a>0?-r:r,h=Math.max(0,-(a*d+o)),f=-h*h+d*(d+2*c)+l;return n&&n.copy(this.origin).addScaledVector(this.direction,h),s&&s.copy(Cl).addScaledVector(ca,d),f}intersectSphere(t,e){if(t.radius<0)return null;$n.subVectors(t.center,this.origin);let n=$n.dot(this.direction),s=$n.dot($n)-n*n,r=t.radius*t.radius;if(s>r)return null;let a=Math.sqrt(r-s),o=n-a,c=n+a;return c<0?null:o<0?this.at(c,e):this.at(o,e)}intersectsSphere(t){return t.radius<0?!1:this.distanceSqToPoint(t.center)<=t.radius*t.radius}distanceToPlane(t){let e=t.normal.dot(this.direction);if(e===0)return t.distanceToPoint(this.origin)===0?0:null;let n=-(this.origin.dot(t.normal)+t.constant)/e;return n>=0?n:null}intersectPlane(t,e){let n=this.distanceToPlane(t);return n===null?null:this.at(n,e)}intersectsPlane(t){let e=t.distanceToPoint(this.origin);return e===0||t.normal.dot(this.direction)*e<0}intersectBox(t,e){let n,s,r,a,o,c,l=1/this.direction.x,u=1/this.direction.y,h=1/this.direction.z,d=this.origin;return l>=0?(n=(t.min.x-d.x)*l,s=(t.max.x-d.x)*l):(n=(t.max.x-d.x)*l,s=(t.min.x-d.x)*l),u>=0?(r=(t.min.y-d.y)*u,a=(t.max.y-d.y)*u):(r=(t.max.y-d.y)*u,a=(t.min.y-d.y)*u),n>a||r>s||((r>n||isNaN(n))&&(n=r),(a<s||isNaN(s))&&(s=a),h>=0?(o=(t.min.z-d.z)*h,c=(t.max.z-d.z)*h):(o=(t.max.z-d.z)*h,c=(t.min.z-d.z)*h),n>c||o>s)||((o>n||n!==n)&&(n=o),(c<s||s!==s)&&(s=c),s<0)?null:this.at(n>=0?n:s,e)}intersectsBox(t){return this.intersectBox(t,$n)!==null}intersectTriangle(t,e,n,s,r){let a=this.origin,o=this.direction,c=o.x,l=o.y,u=o.z,h=t.x-a.x,d=t.y-a.y,f=t.z-a.z,m=e.x-a.x,y=e.y-a.y,g=e.z-a.z,p=n.x-a.x,v=n.y-a.y,b=n.z-a.z,_=Math.abs(c),T=Math.abs(l),w=Math.abs(u),x,M,P,C,k,O,j,D,K,Q,Y,J;if(_>=T&&_>=w?(P=c,O=h,K=m,J=p,c>=0?(x=l,M=u,C=d,k=f,j=y,D=g,Q=v,Y=b):(x=u,M=l,C=f,k=d,j=g,D=y,Q=b,Y=v)):T>=w?(P=l,O=d,K=y,J=v,l>=0?(x=u,M=c,C=f,k=h,j=g,D=m,Q=b,Y=p):(x=c,M=u,C=h,k=f,j=m,D=g,Q=p,Y=b)):(P=u,O=f,K=g,J=b,u>=0?(x=c,M=l,C=h,k=d,j=m,D=y,Q=p,Y=v):(x=l,M=c,C=d,k=h,j=y,D=m,Q=v,Y=p)),P===0)return null;let at=x/P,W=M/P,tt=1/P,vt=C-at*O,wt=k-W*O,st=j-at*K,U=D-W*K,et=Q-at*J,V=Y-W*J,N=et*U-V*st,L=vt*V-wt*et,I=st*wt-U*vt;if(s){if(N<0||L<0||I<0)return null}else if((N<0||L<0||I<0)&&(N>0||L>0||I>0))return null;let G=N+L+I;if(G===0)return null;let ht=tt*(N*O+L*K+I*J);return(G>0?ht<0:ht>0)?null:this.at(ht/G,r)}applyMatrix4(t){return this.origin.applyMatrix4(t),this.direction.transformDirection(t),this}equals(t){return t.origin.equals(this.origin)&&t.direction.equals(this.direction)}clone(){return new this.constructor().copy(this)}},Ae=class extends ti{constructor(t){super(),this.isMeshBasicMaterial=!0,this.type="MeshBasicMaterial",this.color=new Vt(16777215),this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.specularMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new jn,this.combine=ec,this.reflectivity=1,this.refractionRatio=.98,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.specularMap=t.specularMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.envMapRotation.copy(t.envMapRotation),this.combine=t.combine,this.reflectivity=t.reflectivity,this.refractionRatio=t.refractionRatio,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.fog=t.fog,this}},Mh=new xe,Ni=new xi,ua=new Qn,Sh=new F,da=new F,fa=new F,pa=new F,Rl=new F,ma=new F,wh=new F,ga=new F,he=class extends Ne{constructor(t=new de,e=new Ae){super(),this.isMesh=!0,this.type="Mesh",this.geometry=t,this.material=e,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.count=1,this.updateMorphTargets()}copy(t,e){return super.copy(t,e),t.morphTargetInfluences!==void 0?this.morphTargetInfluences=t.morphTargetInfluences.slice():this.morphTargetInfluences=void 0,t.morphTargetDictionary!==void 0?this.morphTargetDictionary=Object.assign({},t.morphTargetDictionary):this.morphTargetDictionary=void 0,this.material=Array.isArray(t.material)?t.material.slice():t.material,this.geometry=t.geometry,this}updateMorphTargets(){let e=this.geometry.morphAttributes,n=Object.keys(e);if(n.length>0){let s=e[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){let o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}getVertexPosition(t,e){let n=this.geometry,s=n.attributes.position,r=n.morphAttributes.position,a=n.morphTargetsRelative;e.fromBufferAttribute(s,t);let o=this.morphTargetInfluences;if(r&&o){ma.set(0,0,0);for(let c=0,l=r.length;c<l;c++){let u=o[c],h=r[c];u!==0&&(Rl.fromBufferAttribute(h,t),a?ma.addScaledVector(Rl,u):ma.addScaledVector(Rl.sub(e),u))}e.add(ma)}return e}intersectsFrustum(t){return t.intersectsObject(this)}raycast(t,e){let n=this.geometry,s=this.material,r=this.matrixWorld;s!==void 0&&(n.boundingSphere===null&&n.computeBoundingSphere(),ua.copy(n.boundingSphere),ua.applyMatrix4(r),Ni.copy(t.ray).recast(t.near),!(ua.containsPoint(Ni.origin)===!1&&(Ni.intersectSphere(ua,Sh)===null||Ni.origin.distanceToSquared(Sh)>(t.far-t.near)**2))&&(Mh.copy(r).invert(),Ni.copy(t.ray).applyMatrix4(Mh),!(n.boundingBox!==null&&Ni.intersectsBox(n.boundingBox)===!1)&&this._computeIntersections(t,e,Ni)))}_computeIntersections(t,e,n){let s,r=this.geometry,a=this.material,o=r.index,c=r.attributes.position,l=r.attributes.uv,u=r.attributes.uv1,h=r.attributes.normal,d=r.groups,f=r.drawRange;if(o!==null)if(Array.isArray(a))for(let m=0,y=d.length;m<y;m++){let g=d[m],p=a[g.materialIndex],v=Math.max(g.start,f.start),b=Math.min(o.count,Math.min(g.start+g.count,f.start+f.count));for(let _=v,T=b;_<T;_+=3){let w=o.getX(_),x=o.getX(_+1),M=o.getX(_+2);s=xa(this,p,t,n,l,u,h,w,x,M),s&&(s.faceIndex=Math.floor(_/3),s.face.materialIndex=g.materialIndex,e.push(s))}}else{let m=Math.max(0,f.start),y=Math.min(o.count,f.start+f.count);for(let g=m,p=y;g<p;g+=3){let v=o.getX(g),b=o.getX(g+1),_=o.getX(g+2);s=xa(this,a,t,n,l,u,h,v,b,_),s&&(s.faceIndex=Math.floor(g/3),e.push(s))}}else if(c!==void 0)if(Array.isArray(a))for(let m=0,y=d.length;m<y;m++){let g=d[m],p=a[g.materialIndex],v=Math.max(g.start,f.start),b=Math.min(c.count,Math.min(g.start+g.count,f.start+f.count));for(let _=v,T=b;_<T;_+=3){let w=_,x=_+1,M=_+2;s=xa(this,p,t,n,l,u,h,w,x,M),s&&(s.faceIndex=Math.floor(_/3),s.face.materialIndex=g.materialIndex,e.push(s))}}else{let m=Math.max(0,f.start),y=Math.min(c.count,f.start+f.count);for(let g=m,p=y;g<p;g+=3){let v=g,b=g+1,_=g+2;s=xa(this,a,t,n,l,u,h,v,b,_),s&&(s.faceIndex=Math.floor(g/3),e.push(s))}}}};function kf(i,t,e,n,s,r,a,o){let c;if(t.side===We?c=n.intersectTriangle(a,r,s,!0,o):c=n.intersectTriangle(s,r,a,t.side===wi,o),c===null)return null;ga.copy(o),ga.applyMatrix4(i.matrixWorld);let l=e.ray.origin.distanceTo(ga);return l<e.near||l>e.far?null:{distance:l,point:ga.clone(),object:i}}function xa(i,t,e,n,s,r,a,o,c,l){i.getVertexPosition(o,da),i.getVertexPosition(c,fa),i.getVertexPosition(l,pa);let u=kf(i,t,e,n,da,fa,pa,wh);if(u){let h=new F;gi.getBarycoord(wh,da,fa,pa,h),s&&(u.uv=gi.getInterpolatedAttribute(s,o,c,l,h,new ft)),r&&(u.uv1=gi.getInterpolatedAttribute(r,o,c,l,h,new ft)),a&&(u.normal=gi.getInterpolatedAttribute(a,o,c,l,h,new F),u.normal.dot(n.direction)>0&&u.normal.multiplyScalar(-1));let d={a:o,b:c,c:l,normal:new F,materialIndex:0};gi.getNormal(da,fa,pa,d.normal),u.face=d,u.barycoord=h}return u}var hr=class extends Qe{constructor(t=null,e=1,n=1,s,r,a,o,c,l=Ge,u=Ge,h,d){super(null,a,o,c,l,u,s,r,h,d),this.isDataTexture=!0,this.image={data:t,width:e,height:n},this.generateMipmaps=!1,this.flipY=!1,this.unpackAlignment=1}};var ur=class extends Mn{constructor(t,e,n,s=1){super(t,e,n),this.isInstancedBufferAttribute=!0,this.meshPerAttribute=s}copy(t){return super.copy(t),this.meshPerAttribute=t.meshPerAttribute,this}toJSON(){let t=super.toJSON();return t.meshPerAttribute=this.meshPerAttribute,t.isInstancedBufferAttribute=!0,t}},cs=new xe,Th=new xe,_a=[],Ah=new Fn,zf=new xe,Ys=new he,$s=new Qn,dr=class extends he{constructor(t,e,n){super(t,e),this.isInstancedMesh=!0,this.instanceMatrix=new ur(new Float32Array(n*16),16),this.instanceColor=null,this.morphTexture=null,this.count=n,this.boundingBox=null,this.boundingSphere=null;for(let s=0;s<n;s++)this.setMatrixAt(s,zf)}computeBoundingBox(){let t=this.geometry,e=this.count;this.boundingBox===null&&(this.boundingBox=new Fn),t.boundingBox===null&&t.computeBoundingBox(),this.boundingBox.makeEmpty();for(let n=0;n<e;n++)this.getMatrixAt(n,cs),Ah.copy(t.boundingBox).applyMatrix4(cs),this.boundingBox.union(Ah)}computeBoundingSphere(){let t=this.geometry,e=this.count;this.boundingSphere===null&&(this.boundingSphere=new Qn),t.boundingSphere===null&&t.computeBoundingSphere(),this.boundingSphere.makeEmpty();for(let n=0;n<e;n++)this.getMatrixAt(n,cs),$s.copy(t.boundingSphere).applyMatrix4(cs),this.boundingSphere.union($s)}copy(t,e){return super.copy(t,e),this.instanceMatrix.copy(t.instanceMatrix),this.morphTexture=t.morphTexture!==null?t.morphTexture.clone():null,this.instanceColor=t.instanceColor!==null?t.instanceColor.clone():null,this.count=t.count,this.boundingBox=t.boundingBox!==null?t.boundingBox.clone():null,this.boundingSphere=t.boundingSphere!==null?t.boundingSphere.clone():null,this}getColorAt(t,e){return this.instanceColor===null?e.setRGB(1,1,1):e.fromArray(this.instanceColor.array,t*3)}getMatrixAt(t,e){return e.fromArray(this.instanceMatrix.array,t*16)}getMorphAt(t,e){let n=e.morphTargetInfluences,s=this.morphTexture.source.data.data,r=n.length+1,a=t*r+1;for(let o=0;o<n.length;o++)n[o]=s[a+o]}raycast(t,e){let n=this.matrixWorld,s=this.count;if(Ys.geometry=this.geometry,Ys.material=this.material,Ys.material!==void 0&&(this.boundingSphere===null&&this.computeBoundingSphere(),$s.copy(this.boundingSphere),$s.applyMatrix4(n),t.ray.intersectsSphere($s)!==!1))for(let r=0;r<s;r++){this.getMatrixAt(r,cs),Th.multiplyMatrices(n,cs),Ys.matrixWorld=Th,Ys.raycast(t,_a);for(let a=0,o=_a.length;a<o;a++){let c=_a[a];c.instanceId=r,c.object=this,e.push(c)}_a.length=0}}setColorAt(t,e){return this.instanceColor===null&&(this.instanceColor=new ur(new Float32Array(this.instanceMatrix.count*3).fill(1),3)),e.toArray(this.instanceColor.array,t*3),this}setMatrixAt(t,e){return e.toArray(this.instanceMatrix.array,t*16),this}setMorphAt(t,e){let n=e.morphTargetInfluences,s=n.length+1;this.morphTexture===null&&(this.morphTexture=new hr(new Float32Array(s*this.count),s,this.count,fo,xn));let r=this.morphTexture.source.data.data,a=0;for(let l=0;l<n.length;l++)a+=n[l];let o=this.geometry.morphTargetsRelative?1:1-a,c=s*t;return r[c]=o,r.set(n,c+1),this}updateMorphTargets(){}dispose(){super.dispose(),this.instanceMatrix.dispose(),this.instanceColor!==null&&this.instanceColor.dispose(),this.morphTexture!==null&&(this.morphTexture.dispose(),this.morphTexture=null)}},Ui=new Qn,Gf=new ft(.5,.5),ya=new F,_i=class{constructor(t=new on,e=new on,n=new on,s=new on,r=new on,a=new on){this.planes=[t,e,n,s,r,a]}set(t,e,n,s,r,a){let o=this.planes;return o[0].copy(t),o[1].copy(e),o[2].copy(n),o[3].copy(s),o[4].copy(r),o[5].copy(a),this}copy(t){let e=this.planes;for(let n=0;n<6;n++)e[n].copy(t.planes[n]);return this}setFromProjectionMatrix(t,e=gn,n=!1){let s=this.planes,r=t.elements,a=r[0],o=r[1],c=r[2],l=r[3],u=r[4],h=r[5],d=r[6],f=r[7],m=r[8],y=r[9],g=r[10],p=r[11],v=r[12],b=r[13],_=r[14],T=r[15];if(s[0].setComponents(l-a,f-u,p-m,T-v).normalize(),s[1].setComponents(l+a,f+u,p+m,T+v).normalize(),s[2].setComponents(l+o,f+h,p+y,T+b).normalize(),s[3].setComponents(l-o,f-h,p-y,T-b).normalize(),n)s[4].setComponents(c,d,g,_).normalize(),s[5].setComponents(l-c,f-d,p-g,T-_).normalize();else if(s[4].setComponents(l-c,f-d,p-g,T-_).normalize(),e===gn)s[5].setComponents(l+c,f+d,p+g,T+_).normalize();else if(e===xs)s[5].setComponents(c,d,g,_).normalize();else throw new Error("THREE.Frustum.setFromProjectionMatrix(): Invalid coordinate system: "+e);return this}intersectsObject(t){if(t.boundingSphere!==void 0)t.boundingSphere===null&&t.computeBoundingSphere(),Ui.copy(t.boundingSphere).applyMatrix4(t.matrixWorld);else{let e=t.geometry;e.boundingSphere===null&&e.computeBoundingSphere(),Ui.copy(e.boundingSphere).applyMatrix4(t.matrixWorld)}return this.intersectsSphere(Ui)}intersectsSprite(t){Ui.center.set(0,0,0);let e=Gf.distanceTo(t.center);return Ui.radius=.7071067811865476+e,Ui.applyMatrix4(t.matrixWorld),this.intersectsSphere(Ui)}intersectsSphere(t){let e=this.planes,n=t.center,s=-t.radius;for(let r=0;r<6;r++)if(e[r].distanceToPoint(n)<s)return!1;return!0}intersectsBox(t){let e=this.planes;for(let n=0;n<6;n++){let s=e[n];if(ya.x=s.normal.x>0?t.max.x:t.min.x,ya.y=s.normal.y>0?t.max.y:t.min.y,ya.z=s.normal.z>0?t.max.z:t.min.z,s.distanceToPoint(ya)<0)return!1}return!0}containsPoint(t){let e=this.planes;for(let n=0;n<6;n++)if(e[n].distanceToPoint(t)<0)return!1;return!0}clone(){return new this.constructor().copy(this)}},Eh=new xe,za=class i{constructor(){this.coordinateSystem=gn,this._frustums=[],this._count=0}setFromArrayCamera(t){let e=t.cameras,n=this._frustums;for(let s=0;s<e.length;s++){let r=e[s];Eh.multiplyMatrices(r.projectionMatrix,r.matrixWorldInverse),n[s]===void 0&&(n[s]=new _i),n[s].setFromProjectionMatrix(Eh,r.coordinateSystem,r.reversedDepth)}return this._count=e.length,this}getFrustum(t){return this._frustums[t]}intersectsObject(t){let e=this._frustums;for(let n=0;n<this._count;n++)if(e[n].intersectsObject(t))return!0;return!1}intersectsSprite(t){let e=this._frustums;for(let n=0;n<this._count;n++)if(e[n].intersectsSprite(t))return!0;return!1}intersectsSphere(t){let e=this._frustums;for(let n=0;n<this._count;n++)if(e[n].intersectsSphere(t))return!0;return!1}intersectsBox(t){let e=this._frustums;for(let n=0;n<this._count;n++)if(e[n].intersectsBox(t))return!0;return!1}containsPoint(t){let e=this._frustums;for(let n=0;n<this._count;n++)if(e[n].containsPoint(t))return!0;return!1}copy(t){this.coordinateSystem=t.coordinateSystem;let e=this._frustums,n=t._frustums;for(let s=0;s<t._count;s++)e[s]===void 0&&(e[s]=new _i),e[s].copy(n[s]);return this._count=t._count,this}clone(){return new i().copy(this)}};var Fi=class extends ti{constructor(t){super(),this.isLineBasicMaterial=!0,this.type="LineBasicMaterial",this.color=new Vt(16777215),this.map=null,this.linewidth=1,this.linecap="round",this.linejoin="round",this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.color.copy(t.color),this.map=t.map,this.linewidth=t.linewidth,this.linecap=t.linecap,this.linejoin=t.linejoin,this.fog=t.fog,this}},Ga=new F,Va=new F,Ch=new xe,Zs=new xi,va=new Qn,Pl=new F,Rh=new F,Ss=class extends Ne{constructor(t=new de,e=new Fi){super(),this.isLine=!0,this.type="Line",this.geometry=t,this.material=e,this.morphTargetDictionary=void 0,this.morphTargetInfluences=void 0,this.updateMorphTargets()}copy(t,e){return super.copy(t,e),t.morphTargetInfluences!==void 0?this.morphTargetInfluences=t.morphTargetInfluences.slice():this.morphTargetInfluences=void 0,t.morphTargetDictionary!==void 0?this.morphTargetDictionary=Object.assign({},t.morphTargetDictionary):this.morphTargetDictionary=void 0,this.material=Array.isArray(t.material)?t.material.slice():t.material,this.geometry=t.geometry,this}computeLineDistances(){let t=this.geometry;if(t.index===null){let e=t.attributes.position,n=[0];for(let s=1,r=e.count;s<r;s++)Ga.fromBufferAttribute(e,s-1),Va.fromBufferAttribute(e,s),n[s]=n[s-1],n[s]+=Ga.distanceTo(Va);t.setAttribute("lineDistance",new Jt(n,1))}else qt("Line.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}intersectsFrustum(t){return t.intersectsObject(this)}raycast(t,e){let n=this.geometry,s=this.matrixWorld,r=t.params.Line.threshold,a=n.drawRange;if(n.boundingSphere===null&&n.computeBoundingSphere(),va.copy(n.boundingSphere),va.applyMatrix4(s),va.radius+=r,t.ray.intersectsSphere(va)===!1)return;Ch.copy(s).invert(),Zs.copy(t.ray).applyMatrix4(Ch);let o=r/((this.scale.x+this.scale.y+this.scale.z)/3),c=o*o,l=this.isLineSegments?2:1,u=n.index,d=n.attributes.position;if(u!==null){let f=Math.max(0,a.start),m=Math.min(u.count,a.start+a.count);for(let y=f,g=m-1;y<g;y+=l){let p=u.getX(y),v=u.getX(y+1),b=ba(this,t,Zs,c,p,v,y);b&&e.push(b)}if(this.isLineLoop){let y=u.getX(m-1),g=u.getX(f),p=ba(this,t,Zs,c,y,g,m-1);p&&e.push(p)}}else{let f=Math.max(0,a.start),m=Math.min(d.count,a.start+a.count);for(let y=f,g=m-1;y<g;y+=l){let p=ba(this,t,Zs,c,y,y+1,y);p&&e.push(p)}if(this.isLineLoop){let y=ba(this,t,Zs,c,m-1,f,m-1);y&&e.push(y)}}}updateMorphTargets(){let e=this.geometry.morphAttributes,n=Object.keys(e);if(n.length>0){let s=e[n[0]];if(s!==void 0){this.morphTargetInfluences=[],this.morphTargetDictionary={};for(let r=0,a=s.length;r<a;r++){let o=s[r].name||String(r);this.morphTargetInfluences.push(0),this.morphTargetDictionary[o]=r}}}}};function ba(i,t,e,n,s,r,a){let o=i.geometry.attributes.position;if(Ga.fromBufferAttribute(o,s),Va.fromBufferAttribute(o,r),e.distanceSqToSegment(Ga,Va,Pl,Rh)>n)return;Pl.applyMatrix4(i.matrixWorld);let l=t.ray.origin.distanceTo(Pl);if(!(l<t.near||l>t.far))return{distance:l,point:Rh.clone().applyMatrix4(i.matrixWorld),index:a,face:null,faceIndex:null,barycoord:null,object:i}}var Ph=new F,Ih=new F,ws=class extends Ss{constructor(t,e){super(t,e),this.isLineSegments=!0,this.type="LineSegments"}computeLineDistances(){let t=this.geometry;if(t.index===null){let e=t.attributes.position,n=[];for(let s=0,r=e.count;s<r;s+=2)Ph.fromBufferAttribute(e,s),Ih.fromBufferAttribute(e,s+1),n[s]=s===0?0:n[s-1],n[s+1]=n[s]+Ph.distanceTo(Ih);t.setAttribute("lineDistance",new Jt(n,1))}else qt("LineSegments.computeLineDistances(): Computation only possible with non-indexed BufferGeometry.");return this}};var fr=class extends Qe{constructor(t=[],e=Yi,n,s,r,a,o,c,l,u){super(t,e,n,s,r,a,o,c,l,u),this.isCubeTexture=!0,this.flipY=!1}get images(){return this.image}set images(t){this.image=t}},Oi=class extends Qe{constructor(t,e,n,s,r,a,o,c,l){super(t,e,n,s,r,a,o,c,l),this.isCanvasTexture=!0,this.needsUpdate=!0}};var ei=class extends Qe{constructor(t,e,n=In,s,r,a,o=Ge,c=Ge,l,u=Kn,h=1){if(u!==Kn&&u!==ri)throw new Error("THREE.DepthTexture: format must be either THREE.DepthFormat or THREE.DepthStencilFormat");let d={width:t,height:e,depth:h};super(d,s,r,a,o,c,u,n,l),this.isDepthTexture=!0,this.flipY=!1,this.generateMipmaps=!1,this.compareFunction=null}copy(t){return super.copy(t),this.source=new vs(Object.assign({},t.image)),this.compareFunction=t.compareFunction,this}toJSON(t){let e=super.toJSON(t);return e.compareFunction=this.compareFunction,e}},pr=class extends ei{constructor(t,e=In,n=Yi,s,r,a=Ge,o=Ge,c,l=Kn){let u={width:t,height:t,depth:1},h=[u,u,u,u,u,u];super(t,t,e,n,s,r,a,o,c,l),this.image=h,this.isCubeDepthTexture=!0,this.isCubeTexture=!0}get images(){return this.image}set images(t){this.image=t}},mr=class extends Qe{constructor(t=null){super(),this.sourceTexture=t,this.isExternalTexture=!0}copy(t){return super.copy(t),this.sourceTexture=t.sourceTexture,this}},Ce=class i extends de{constructor(t=1,e=1,n=1,s=1,r=1,a=1){super(),this.type="BoxGeometry",this.parameters={width:t,height:e,depth:n,widthSegments:s,heightSegments:r,depthSegments:a};let o=this;s=Math.floor(s),r=Math.floor(r),a=Math.floor(a);let c=[],l=[],u=[],h=[],d=0,f=0;m("z","y","x",-1,-1,n,e,t,a,r,0),m("z","y","x",1,-1,n,e,-t,a,r,1),m("x","z","y",1,1,t,n,e,s,a,2),m("x","z","y",1,-1,t,n,-e,s,a,3),m("x","y","z",1,-1,t,e,n,s,r,4),m("x","y","z",-1,-1,t,e,-n,s,r,5),this.setIndex(c),this.setAttribute("position",new Jt(l,3)),this.setAttribute("normal",new Jt(u,3)),this.setAttribute("uv",new Jt(h,2));function m(y,g,p,v,b,_,T,w,x,M,P){let C=_/x,k=T/M,O=_/2,j=T/2,D=w/2,K=x+1,Q=M+1,Y=0,J=0,at=new F;for(let W=0;W<Q;W++){let tt=W*k-j;for(let vt=0;vt<K;vt++){let wt=vt*C-O;at[y]=wt*v,at[g]=tt*b,at[p]=D,l.push(at.x,at.y,at.z),at[y]=0,at[g]=0,at[p]=w>0?1:-1,u.push(at.x,at.y,at.z),h.push(vt/x),h.push(1-W/M),Y+=1}}for(let W=0;W<M;W++)for(let tt=0;tt<x;tt++){let vt=d+tt+K*W,wt=d+tt+K*(W+1),st=d+(tt+1)+K*(W+1),U=d+(tt+1)+K*W;c.push(vt,wt,U),c.push(wt,st,U),J+=6}o.addGroup(f,J,P),f+=J,d+=Y}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new i(t.width,t.height,t.depth,t.widthSegments,t.heightSegments,t.depthSegments)}},Bi=class i extends de{constructor(t=1,e=1,n=4,s=8,r=1){super(),this.type="CapsuleGeometry",this.parameters={radius:t,height:e,capSegments:n,radialSegments:s,heightSegments:r},e=Math.max(0,e),n=Math.max(1,Math.floor(n)),s=Math.max(3,Math.floor(s)),r=Math.max(1,Math.floor(r));let a=[],o=[],c=[],l=[],u=e/2,h=Math.PI/2*t,d=e,f=2*h+d,m=n*2+r,y=s+1,g=new F,p=new F;for(let v=0;v<=m;v++){let b=0,_=0,T=0,w=0;if(v<=n){let P=v/n,C=P*Math.PI/2;_=-u-t*Math.cos(C),T=t*Math.sin(C),w=-t*Math.cos(C),b=P*h}else if(v<=n+r){let P=(v-n)/r;_=-u+P*e,T=t,w=0,b=h+P*d}else{let P=(v-n-r)/n,C=P*Math.PI/2;_=u+t*Math.sin(C),T=t*Math.cos(C),w=t*Math.sin(C),b=h+d+P*h}let x=Math.max(0,Math.min(1,b/f)),M=0;v===0?M=.5/s:v===m&&(M=-.5/s);for(let P=0;P<=s;P++){let C=P/s,k=C*Math.PI*2,O=Math.sin(k),j=Math.cos(k);p.x=-T*j,p.y=_,p.z=T*O,o.push(p.x,p.y,p.z),g.set(-T*j,w,T*O),g.normalize(),c.push(g.x,g.y,g.z),l.push(C+M,x)}if(v>0){let P=(v-1)*y;for(let C=0;C<s;C++){let k=P+C,O=P+C+1,j=v*y+C,D=v*y+C+1;a.push(k,O,j),a.push(O,D,j)}}}this.setIndex(a),this.setAttribute("position",new Jt(o,3)),this.setAttribute("normal",new Jt(c,3)),this.setAttribute("uv",new Jt(l,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new i(t.radius,t.height,t.capSegments,t.radialSegments,t.heightSegments)}},ki=class i extends de{constructor(t=1,e=32,n=0,s=Math.PI*2){super(),this.type="CircleGeometry",this.parameters={radius:t,segments:e,thetaStart:n,thetaLength:s},e=Math.max(3,e);let r=[],a=[],o=[],c=[],l=new F,u=new ft;a.push(0,0,0),o.push(0,0,1),c.push(.5,.5);for(let h=0,d=3;h<=e;h++,d+=3){let f=n+h/e*s;l.x=t*Math.cos(f),l.y=t*Math.sin(f),a.push(l.x,l.y,l.z),o.push(0,0,1),u.x=(a[d]/t+1)/2,u.y=(a[d+1]/t+1)/2,c.push(u.x,u.y)}for(let h=1;h<=e;h++)r.push(h,h+1,0);this.setIndex(r),this.setAttribute("position",new Jt(a,3)),this.setAttribute("normal",new Jt(o,3)),this.setAttribute("uv",new Jt(c,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new i(t.radius,t.segments,t.thetaStart,t.thetaLength)}},Le=class i extends de{constructor(t=1,e=1,n=1,s=32,r=1,a=!1,o=0,c=Math.PI*2){super(),this.type="CylinderGeometry",this.parameters={radiusTop:t,radiusBottom:e,height:n,radialSegments:s,heightSegments:r,openEnded:a,thetaStart:o,thetaLength:c};let l=this;s=Math.floor(s),r=Math.floor(r);let u=[],h=[],d=[],f=[],m=0,y=[],g=n/2,p=0;v(),a===!1&&(t>0&&b(!0),e>0&&b(!1)),this.setIndex(u),this.setAttribute("position",new Jt(h,3)),this.setAttribute("normal",new Jt(d,3)),this.setAttribute("uv",new Jt(f,2));function v(){let _=new F,T=new F,w=0,x=(e-t)/n;for(let M=0;M<=r;M++){let P=[],C=M/r,k=C*(e-t)+t;for(let O=0;O<=s;O++){let j=O/s,D=j*c+o,K=Math.sin(D),Q=Math.cos(D);T.x=k*K,T.y=-C*n+g,T.z=k*Q,h.push(T.x,T.y,T.z),_.set(K,x,Q).normalize(),d.push(_.x,_.y,_.z),f.push(j,1-C),P.push(m++)}y.push(P)}for(let M=0;M<s;M++)for(let P=0;P<r;P++){let C=y[P][M],k=y[P+1][M],O=y[P+1][M+1],j=y[P][M+1];(t>0||P!==0)&&(u.push(C,k,j),w+=3),(e>0||P!==r-1)&&(u.push(k,O,j),w+=3)}l.addGroup(p,w,0),p+=w}function b(_){let T=m,w=new ft,x=new F,M=0,P=_===!0?t:e,C=_===!0?1:-1;for(let O=1;O<=s;O++)h.push(0,g*C,0),d.push(0,C,0),f.push(.5,.5),m++;let k=m;for(let O=0;O<=s;O++){let D=O/s*c+o,K=Math.cos(D),Q=Math.sin(D);x.x=P*Q,x.y=g*C,x.z=P*K,h.push(x.x,x.y,x.z),d.push(0,C,0),w.x=K*.5+.5,w.y=Q*.5*C+.5,f.push(w.x,w.y),m++}for(let O=0;O<s;O++){let j=T+O,D=k+O;_===!0?u.push(D,D+1,j):u.push(D+1,D,j),M+=3}l.addGroup(p,M,_===!0?1:2),p+=M}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new i(t.radiusTop,t.radiusBottom,t.height,t.radialSegments,t.heightSegments,t.openEnded,t.thetaStart,t.thetaLength)}},Tn=class i extends Le{constructor(t=1,e=1,n=32,s=1,r=!1,a=0,o=Math.PI*2){super(0,t,e,n,s,r,a,o),this.type="ConeGeometry",this.parameters={radius:t,height:e,radialSegments:n,heightSegments:s,openEnded:r,thetaStart:a,thetaLength:o}}static fromJSON(t){return new i(t.radius,t.height,t.radialSegments,t.heightSegments,t.openEnded,t.thetaStart,t.thetaLength)}},gr=class i extends de{constructor(t=[],e=[],n=1,s=0){super(),this.type="PolyhedronGeometry",this.parameters={vertices:t,indices:e,radius:n,detail:s};let r=[],a=[];o(s),l(n),u(),this.setAttribute("position",new Jt(r,3)),this.setAttribute("normal",new Jt(r.slice(),3)),this.setAttribute("uv",new Jt(a,2)),s===0?this.computeVertexNormals():this.normalizeNormals();function o(v){let b=new F,_=new F,T=new F;for(let w=0;w<e.length;w+=3)f(e[w+0],b),f(e[w+1],_),f(e[w+2],T),c(b,_,T,v)}function c(v,b,_,T){let w=T+1,x=[];for(let M=0;M<=w;M++){x[M]=[];let P=v.clone().lerp(_,M/w),C=b.clone().lerp(_,M/w),k=w-M;for(let O=0;O<=k;O++)O===0&&M===w?x[M][O]=P:x[M][O]=P.clone().lerp(C,O/k)}for(let M=0;M<w;M++)for(let P=0;P<2*(w-M)-1;P++){let C=Math.floor(P/2);P%2===0?(d(x[M][C+1]),d(x[M+1][C]),d(x[M][C])):(d(x[M][C+1]),d(x[M+1][C+1]),d(x[M+1][C]))}}function l(v){let b=new F;for(let _=0;_<r.length;_+=3)b.x=r[_+0],b.y=r[_+1],b.z=r[_+2],b.normalize().multiplyScalar(v),r[_+0]=b.x,r[_+1]=b.y,r[_+2]=b.z}function u(){let v=new F;for(let b=0;b<r.length;b+=3){v.x=r[b+0],v.y=r[b+1],v.z=r[b+2];let _=g(v)/2/Math.PI+.5,T=p(v)/Math.PI+.5;a.push(_,1-T)}m(),h()}function h(){for(let v=0;v<a.length;v+=6){let b=a[v+0],_=a[v+2],T=a[v+4],w=Math.max(b,_,T),x=Math.min(b,_,T);w>.9&&x<.1&&(b<.2&&(a[v+0]+=1),_<.2&&(a[v+2]+=1),T<.2&&(a[v+4]+=1))}}function d(v){r.push(v.x,v.y,v.z)}function f(v,b){let _=v*3;b.x=t[_+0],b.y=t[_+1],b.z=t[_+2]}function m(){let v=new F,b=new F,_=new F,T=new F,w=new ft,x=new ft,M=new ft;for(let P=0,C=0;P<r.length;P+=9,C+=6){v.set(r[P+0],r[P+1],r[P+2]),b.set(r[P+3],r[P+4],r[P+5]),_.set(r[P+6],r[P+7],r[P+8]),w.set(a[C+0],a[C+1]),x.set(a[C+2],a[C+3]),M.set(a[C+4],a[C+5]),T.copy(v).add(b).add(_).divideScalar(3);let k=g(T);y(w,C+0,v,k),y(x,C+2,b,k),y(M,C+4,_,k)}}function y(v,b,_,T){T<0&&v.x===1&&(a[b]=v.x-1),_.x===0&&_.z===0&&(a[b]=T/2/Math.PI+.5)}function g(v){return Math.atan2(v.z,-v.x)}function p(v){return Math.atan2(-v.y,Math.sqrt(v.x*v.x+v.z*v.z))}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new i(t.vertices,t.indices,t.radius,t.detail)}},xr=class i extends gr{constructor(t=1,e=0){let n=(1+Math.sqrt(5))/2,s=1/n,r=[-1,-1,-1,-1,-1,1,-1,1,-1,-1,1,1,1,-1,-1,1,-1,1,1,1,-1,1,1,1,0,-s,-n,0,-s,n,0,s,-n,0,s,n,-s,-n,0,-s,n,0,s,-n,0,s,n,0,-n,0,-s,n,0,-s,-n,0,s,n,0,s],a=[3,11,7,3,7,15,3,15,13,7,19,17,7,17,6,7,6,15,17,4,8,17,8,10,17,10,6,8,0,16,8,16,2,8,2,10,0,12,1,0,1,18,0,18,16,6,10,2,6,2,13,6,13,15,2,16,18,2,18,3,2,3,13,18,1,9,18,9,11,18,11,3,4,14,12,4,12,0,4,0,8,11,9,5,11,5,19,11,19,7,19,5,14,19,14,4,19,4,17,1,12,14,1,14,5,1,5,9];super(r,a,t,e),this.type="DodecahedronGeometry",this.parameters={radius:t,detail:e}}static fromJSON(t){return new i(t.radius,t.detail)}};var hn=class{constructor(){this.type="Curve",this.arcLengthDivisions=200,this.needsUpdate=!1,this.cacheArcLengths=null}getPoint(){qt("Curve: .getPoint() not implemented.")}getPointAt(t,e){let n=this.getUtoTmapping(t);return this.getPoint(n,e)}getPoints(t=5){let e=[];for(let n=0;n<=t;n++)e.push(this.getPoint(n/t));return e}getSpacedPoints(t=5){let e=[];for(let n=0;n<=t;n++)e.push(this.getPointAt(n/t));return e}getLength(){let t=this.getLengths();return t[t.length-1]}getLengths(t=this.arcLengthDivisions){if(this.cacheArcLengths&&this.cacheArcLengths.length===t+1&&!this.needsUpdate)return this.cacheArcLengths;this.needsUpdate=!1;let e=[],n,s=this.getPoint(0),r=0;e.push(0);for(let a=1;a<=t;a++)n=this.getPoint(a/t),r+=n.distanceTo(s),e.push(r),s=n;return this.cacheArcLengths=e,e}updateArcLengths(){this.needsUpdate=!0,this.getLengths()}getUtoTmapping(t,e=null){let n=this.getLengths(),s=0,r=n.length,a;e?a=e:a=t*n[r-1];let o=0,c=r-1,l;for(;o<=c;)if(s=Math.floor(o+(c-o)/2),l=n[s]-a,l<0)o=s+1;else if(l>0)c=s-1;else{c=s;break}if(s=c,n[s]===a)return s/(r-1);let u=n[s],d=n[s+1]-u,f=(a-u)/d;return(s+f)/(r-1)}getTangent(t,e){let s=t-1e-4,r=t+1e-4;s<0&&(s=0),r>1&&(r=1);let a=this.getPoint(s),o=this.getPoint(r),c=e||(a.isVector2?new ft:new F);return c.copy(o).sub(a).normalize(),c}getTangentAt(t,e){let n=this.getUtoTmapping(t);return this.getTangent(n,e)}computeFrenetFrames(t,e=!1){let n=new F,s=[],r=[],a=[],o=new F,c=new xe;for(let f=0;f<=t;f++){let m=f/t;s[f]=this.getTangentAt(m,new F)}r[0]=new F,a[0]=new F;let l=Number.MAX_VALUE,u=Math.abs(s[0].x),h=Math.abs(s[0].y),d=Math.abs(s[0].z);u<=l&&(l=u,n.set(1,0,0)),h<=l&&(l=h,n.set(0,1,0)),d<=l&&n.set(0,0,1),o.crossVectors(s[0],n).normalize(),r[0].crossVectors(s[0],o),a[0].crossVectors(s[0],r[0]);for(let f=1;f<=t;f++){if(r[f]=r[f-1].clone(),a[f]=a[f-1].clone(),o.crossVectors(s[f-1],s[f]),o.length()>Number.EPSILON){o.normalize();let m=Math.acos(ne(s[f-1].dot(s[f]),-1,1));r[f].applyMatrix4(c.makeRotationAxis(o,m))}a[f].crossVectors(s[f],r[f])}if(e===!0){let f=Math.acos(ne(r[0].dot(r[t]),-1,1));f/=t,s[0].dot(o.crossVectors(r[0],r[t]))>0&&(f=-f);for(let m=1;m<=t;m++)r[m].applyMatrix4(c.makeRotationAxis(s[m],f*m)),a[m].crossVectors(s[m],r[m])}return{tangents:s,normals:r,binormals:a}}clone(){return new this.constructor().copy(this)}copy(t){return this.arcLengthDivisions=t.arcLengthDivisions,this.cacheArcLengths=null,this}toJSON(){let t={metadata:{version:4.7,type:"Curve",generator:"Curve.toJSON"}};return t.arcLengthDivisions=this.arcLengthDivisions,t.type=this.type,t}fromJSON(t){return this.arcLengthDivisions=t.arcLengthDivisions,this.cacheArcLengths=null,this}},Ts=class extends hn{constructor(t=0,e=0,n=1,s=1,r=0,a=Math.PI*2,o=!1,c=0){super(),this.isEllipseCurve=!0,this.type="EllipseCurve",this.aX=t,this.aY=e,this.xRadius=n,this.yRadius=s,this.aStartAngle=r,this.aEndAngle=a,this.aClockwise=o,this.aRotation=c}getPoint(t,e=new ft){let n=e,s=Math.PI*2,r=this.aEndAngle-this.aStartAngle,a=Math.abs(r)<Number.EPSILON;for(;r<0;)r+=s;for(;r>s;)r-=s;r<Number.EPSILON&&(a?r=0:r=s),this.aClockwise===!0&&!a&&(r===s?r=-s:r=r-s);let o=this.aStartAngle+t*r,c=this.aX+this.xRadius*Math.cos(o),l=this.aY+this.yRadius*Math.sin(o);if(this.aRotation!==0){let u=Math.cos(this.aRotation),h=Math.sin(this.aRotation),d=c-this.aX,f=l-this.aY;c=d*u-f*h+this.aX,l=d*h+f*u+this.aY}return n.set(c,l)}copy(t){return super.copy(t),this.aX=t.aX,this.aY=t.aY,this.xRadius=t.xRadius,this.yRadius=t.yRadius,this.aStartAngle=t.aStartAngle,this.aEndAngle=t.aEndAngle,this.aClockwise=t.aClockwise,this.aRotation=t.aRotation,this}toJSON(){let t=super.toJSON();return t.aX=this.aX,t.aY=this.aY,t.xRadius=this.xRadius,t.yRadius=this.yRadius,t.aStartAngle=this.aStartAngle,t.aEndAngle=this.aEndAngle,t.aClockwise=this.aClockwise,t.aRotation=this.aRotation,t}fromJSON(t){return super.fromJSON(t),this.aX=t.aX,this.aY=t.aY,this.xRadius=t.xRadius,this.yRadius=t.yRadius,this.aStartAngle=t.aStartAngle,this.aEndAngle=t.aEndAngle,this.aClockwise=t.aClockwise,this.aRotation=t.aRotation,this}},Ha=class extends Ts{constructor(t,e,n,s,r,a){super(t,e,n,n,s,r,a),this.isArcCurve=!0,this.type="ArcCurve"}};function Sc(){let i=0,t=0,e=0,n=0;function s(r,a,o,c){i=r,t=o,e=-3*r+3*a-2*o-c,n=2*r-2*a+o+c}return{initCatmullRom:function(r,a,o,c,l){s(a,o,l*(o-r),l*(c-a))},initNonuniformCatmullRom:function(r,a,o,c,l,u,h){let d=(a-r)/l-(o-r)/(l+u)+(o-a)/u,f=(o-a)/u-(c-a)/(u+h)+(c-o)/h;d*=u,f*=u,s(a,o,d,f)},calc:function(r){let a=r*r,o=a*r;return i+t*r+e*a+n*o}}}var Lh=new F,Dh=new F,Il=new Sc,Ll=new Sc,Dl=new Sc,tn=class extends hn{constructor(t=[],e=!1,n="centripetal",s=.5){super(),this.isCatmullRomCurve3=!0,this.type="CatmullRomCurve3",this.points=t,this.closed=e,this.curveType=n,this.tension=s}getPoint(t,e=new F){let n=e,s=this.points,r=s.length,a=(r-(this.closed?0:1))*t,o=Math.floor(a),c=a-o;this.closed?o+=o>0?0:(Math.floor(Math.abs(o)/r)+1)*r:c===0&&o===r-1&&(o=r-2,c=1);let l,u;this.closed||o>0?l=s[(o-1)%r]:(Dh.subVectors(s[0],s[1]).add(s[0]),l=Dh);let h=s[o%r],d=s[(o+1)%r];if(this.closed||o+2<r?u=s[(o+2)%r]:(Lh.subVectors(s[r-1],s[r-2]).add(s[r-1]),u=Lh),this.curveType==="centripetal"||this.curveType==="chordal"){let f=this.curveType==="chordal"?.5:.25,m=Math.pow(l.distanceToSquared(h),f),y=Math.pow(h.distanceToSquared(d),f),g=Math.pow(d.distanceToSquared(u),f);y<1e-4&&(y=1),m<1e-4&&(m=y),g<1e-4&&(g=y),Il.initNonuniformCatmullRom(l.x,h.x,d.x,u.x,m,y,g),Ll.initNonuniformCatmullRom(l.y,h.y,d.y,u.y,m,y,g),Dl.initNonuniformCatmullRom(l.z,h.z,d.z,u.z,m,y,g)}else this.curveType==="catmullrom"&&(Il.initCatmullRom(l.x,h.x,d.x,u.x,this.tension),Ll.initCatmullRom(l.y,h.y,d.y,u.y,this.tension),Dl.initCatmullRom(l.z,h.z,d.z,u.z,this.tension));return n.set(Il.calc(c),Ll.calc(c),Dl.calc(c)),n}copy(t){super.copy(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){let s=t.points[e];this.points.push(s.clone())}return this.closed=t.closed,this.curveType=t.curveType,this.tension=t.tension,this}toJSON(){let t=super.toJSON();t.points=[];for(let e=0,n=this.points.length;e<n;e++){let s=this.points[e];t.points.push(s.toArray())}return t.closed=this.closed,t.curveType=this.curveType,t.tension=this.tension,t}fromJSON(t){super.fromJSON(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){let s=t.points[e];this.points.push(new F().fromArray(s))}return this.closed=t.closed,this.curveType=t.curveType,this.tension=t.tension,this}};function Nh(i,t,e,n,s){let r=(n-t)*.5,a=(s-e)*.5,o=i*i,c=i*o;return(2*e-2*n+r+a)*c+(-3*e+3*n-2*r-a)*o+r*i+e}function Vf(i,t){let e=1-i;return e*e*t}function Hf(i,t){return 2*(1-i)*i*t}function Wf(i,t){return i*i*t}function tr(i,t,e,n){return Vf(i,t)+Hf(i,e)+Wf(i,n)}function Xf(i,t){let e=1-i;return e*e*e*t}function qf(i,t){let e=1-i;return 3*e*e*i*t}function Yf(i,t){return 3*(1-i)*i*i*t}function $f(i,t){return i*i*i*t}function er(i,t,e,n,s){return Xf(i,t)+qf(i,e)+Yf(i,n)+$f(i,s)}var _r=class extends hn{constructor(t=new ft,e=new ft,n=new ft,s=new ft){super(),this.isCubicBezierCurve=!0,this.type="CubicBezierCurve",this.v0=t,this.v1=e,this.v2=n,this.v3=s}getPoint(t,e=new ft){let n=e,s=this.v0,r=this.v1,a=this.v2,o=this.v3;return n.set(er(t,s.x,r.x,a.x,o.x),er(t,s.y,r.y,a.y,o.y)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this.v3.copy(t.v3),this}toJSON(){let t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t.v3=this.v3.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this.v3.fromArray(t.v3),this}},Wa=class extends hn{constructor(t=new F,e=new F,n=new F,s=new F){super(),this.isCubicBezierCurve3=!0,this.type="CubicBezierCurve3",this.v0=t,this.v1=e,this.v2=n,this.v3=s}getPoint(t,e=new F){let n=e,s=this.v0,r=this.v1,a=this.v2,o=this.v3;return n.set(er(t,s.x,r.x,a.x,o.x),er(t,s.y,r.y,a.y,o.y),er(t,s.z,r.z,a.z,o.z)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this.v3.copy(t.v3),this}toJSON(){let t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t.v3=this.v3.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this.v3.fromArray(t.v3),this}},yr=class extends hn{constructor(t=new ft,e=new ft){super(),this.isLineCurve=!0,this.type="LineCurve",this.v1=t,this.v2=e}getPoint(t,e=new ft){let n=e;return t===1?n.copy(this.v2):(n.copy(this.v2).sub(this.v1),n.multiplyScalar(t).add(this.v1)),n}getPointAt(t,e){return this.getPoint(t,e)}getTangent(t,e=new ft){return e.subVectors(this.v2,this.v1).normalize()}getTangentAt(t,e){return this.getTangent(t,e)}copy(t){return super.copy(t),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){let t=super.toJSON();return t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}},As=class extends hn{constructor(t=new F,e=new F){super(),this.isLineCurve3=!0,this.type="LineCurve3",this.v1=t,this.v2=e}getPoint(t,e=new F){let n=e;return t===1?n.copy(this.v2):(n.copy(this.v2).sub(this.v1),n.multiplyScalar(t).add(this.v1)),n}getPointAt(t,e){return this.getPoint(t,e)}getTangent(t,e=new F){return e.subVectors(this.v2,this.v1).normalize()}getTangentAt(t,e){return this.getTangent(t,e)}copy(t){return super.copy(t),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){let t=super.toJSON();return t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}},vr=class extends hn{constructor(t=new ft,e=new ft,n=new ft){super(),this.isQuadraticBezierCurve=!0,this.type="QuadraticBezierCurve",this.v0=t,this.v1=e,this.v2=n}getPoint(t,e=new ft){let n=e,s=this.v0,r=this.v1,a=this.v2;return n.set(tr(t,s.x,r.x,a.x),tr(t,s.y,r.y,a.y)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){let t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}},br=class extends hn{constructor(t=new F,e=new F,n=new F){super(),this.isQuadraticBezierCurve3=!0,this.type="QuadraticBezierCurve3",this.v0=t,this.v1=e,this.v2=n}getPoint(t,e=new F){let n=e,s=this.v0,r=this.v1,a=this.v2;return n.set(tr(t,s.x,r.x,a.x),tr(t,s.y,r.y,a.y),tr(t,s.z,r.z,a.z)),n}copy(t){return super.copy(t),this.v0.copy(t.v0),this.v1.copy(t.v1),this.v2.copy(t.v2),this}toJSON(){let t=super.toJSON();return t.v0=this.v0.toArray(),t.v1=this.v1.toArray(),t.v2=this.v2.toArray(),t}fromJSON(t){return super.fromJSON(t),this.v0.fromArray(t.v0),this.v1.fromArray(t.v1),this.v2.fromArray(t.v2),this}},Mr=class extends hn{constructor(t=[]){super(),this.isSplineCurve=!0,this.type="SplineCurve",this.points=t}getPoint(t,e=new ft){let n=e,s=this.points,r=(s.length-1)*t,a=Math.floor(r),o=r-a,c=s[a===0?a:a-1],l=s[a],u=s[a>s.length-2?s.length-1:a+1],h=s[a>s.length-3?s.length-1:a+2];return n.set(Nh(o,c.x,l.x,u.x,h.x),Nh(o,c.y,l.y,u.y,h.y)),n}copy(t){super.copy(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){let s=t.points[e];this.points.push(s.clone())}return this}toJSON(){let t=super.toJSON();t.points=[];for(let e=0,n=this.points.length;e<n;e++){let s=this.points[e];t.points.push(s.toArray())}return t}fromJSON(t){super.fromJSON(t),this.points=[];for(let e=0,n=t.points.length;e<n;e++){let s=t.points[e];this.points.push(new ft().fromArray(s))}return this}},Xa=Object.freeze({__proto__:null,ArcCurve:Ha,CatmullRomCurve3:tn,CubicBezierCurve:_r,CubicBezierCurve3:Wa,EllipseCurve:Ts,LineCurve:yr,LineCurve3:As,QuadraticBezierCurve:vr,QuadraticBezierCurve3:br,SplineCurve:Mr}),Es=class extends hn{constructor(){super(),this.type="CurvePath",this.curves=[],this.autoClose=!1,this.cacheLengths=null}add(t){this.curves.push(t)}closePath(){let t=this.curves[0].getPoint(0),e=this.curves[this.curves.length-1].getPoint(1);if(!t.equals(e)){let n=t.isVector2===!0?"LineCurve":"LineCurve3";this.curves.push(new Xa[n](e,t))}return this}getPoint(t,e){let n=t*this.getLength(),s=this.getCurveLengths(),r=0;for(;r<s.length;){if(s[r]>=n){let a=s[r]-n,o=this.curves[r],c=o.getLength(),l=c===0?0:1-a/c;return o.getPointAt(l,e)}r++}return null}getLength(){let t=this.getCurveLengths();return t[t.length-1]}updateArcLengths(){this.needsUpdate=!0,this.cacheLengths=null,this.getCurveLengths()}getCurveLengths(){if(this.cacheLengths&&this.cacheLengths.length===this.curves.length)return this.cacheLengths;let t=[],e=0;for(let n=0,s=this.curves.length;n<s;n++)e+=this.curves[n].getLength(),t.push(e);return this.cacheLengths=t,t}getSpacedPoints(t=40){let e=[];for(let n=0;n<=t;n++)e.push(this.getPoint(n/t));return this.autoClose&&e.push(e[0]),e}getPoints(t=12){let e=[],n;for(let s=0,r=this.curves;s<r.length;s++){let a=r[s],o=a.isEllipseCurve?t*2:a.isLineCurve||a.isLineCurve3?1:a.isSplineCurve?t*a.points.length:t,c=a.getPoints(o);for(let l=0;l<c.length;l++){let u=c[l];n&&n.equals(u)||(e.push(u),n=u)}}return this.autoClose&&e.length>1&&!e[e.length-1].equals(e[0])&&e.push(e[0]),e}copy(t){super.copy(t),this.curves=[];for(let e=0,n=t.curves.length;e<n;e++){let s=t.curves[e];this.curves.push(s.clone())}return this.autoClose=t.autoClose,this.cacheLengths=null,this}toJSON(){let t=super.toJSON();t.autoClose=this.autoClose,t.curves=[];for(let e=0,n=this.curves.length;e<n;e++){let s=this.curves[e];t.curves.push(s.toJSON())}return t}fromJSON(t){super.fromJSON(t),this.autoClose=t.autoClose,this.curves=[];for(let e=0,n=t.curves.length;e<n;e++){let s=t.curves[e];this.curves.push(new Xa[s.type]().fromJSON(s))}return this.cacheLengths=null,this}},On=class extends Es{constructor(t){super(),this.type="Path",this.currentPoint=new ft,t&&this.setFromPoints(t)}setFromPoints(t){this.moveTo(t[0].x,t[0].y);for(let e=1,n=t.length;e<n;e++)this.lineTo(t[e].x,t[e].y);return this}moveTo(t,e){return this.currentPoint.set(t,e),this}lineTo(t,e){let n=new yr(this.currentPoint.clone(),new ft(t,e));return this.curves.push(n),this.currentPoint.set(t,e),this}quadraticCurveTo(t,e,n,s){let r=new vr(this.currentPoint.clone(),new ft(t,e),new ft(n,s));return this.curves.push(r),this.currentPoint.set(n,s),this}bezierCurveTo(t,e,n,s,r,a){let o=new _r(this.currentPoint.clone(),new ft(t,e),new ft(n,s),new ft(r,a));return this.curves.push(o),this.currentPoint.set(r,a),this}splineThru(t){let e=[this.currentPoint.clone()].concat(t),n=new Mr(e);return this.curves.push(n),this.currentPoint.copy(t[t.length-1]),this}arc(t,e,n,s,r,a){let o=this.currentPoint.x,c=this.currentPoint.y;return this.absarc(t+o,e+c,n,s,r,a),this}absarc(t,e,n,s,r,a){return this.absellipse(t,e,n,n,s,r,a),this}ellipse(t,e,n,s,r,a,o,c){let l=this.currentPoint.x,u=this.currentPoint.y;return this.absellipse(t+l,e+u,n,s,r,a,o,c),this}absellipse(t,e,n,s,r,a,o,c){let l=new Ts(t,e,n,s,r,a,o,c);if(this.curves.length>0){let h=l.getPoint(0);h.equals(this.currentPoint)||this.lineTo(h.x,h.y)}this.curves.push(l);let u=l.getPoint(1);return this.currentPoint.copy(u),this}copy(t){return super.copy(t),this.currentPoint.copy(t.currentPoint),this}toJSON(){let t=super.toJSON();return t.currentPoint=this.currentPoint.toArray(),t}fromJSON(t){return super.fromJSON(t),this.currentPoint.fromArray(t.currentPoint),this}},zi=class extends On{constructor(t){super(t),this.uuid=Zi(),this.type="Shape",this.holes=[]}getPointsHoles(t){let e=[];for(let n=0,s=this.holes.length;n<s;n++)e[n]=this.holes[n].getPoints(t);return e}extractPoints(t){return{shape:this.getPoints(t),holes:this.getPointsHoles(t)}}copy(t){super.copy(t),this.holes=[];for(let e=0,n=t.holes.length;e<n;e++){let s=t.holes[e];this.holes.push(s.clone())}return this}toJSON(){let t=super.toJSON();t.uuid=this.uuid,t.holes=[];for(let e=0,n=this.holes.length;e<n;e++){let s=this.holes[e];t.holes.push(s.toJSON())}return t}fromJSON(t){super.fromJSON(t),this.uuid=t.uuid,this.holes=[];for(let e=0,n=t.holes.length;e<n;e++){let s=t.holes[e];this.holes.push(new On().fromJSON(s))}return this}};function Zf(i,t,e=2){let n=t&&t.length,s=n?t[0]*e:i.length,r=Nu(i,0,s,e,!0),a=[];if(!r||r.next===r.prev)return a;let o,c,l;if(n&&(r=tp(i,t,r,e)),i.length>80*e){o=i[0],c=i[1];let u=o,h=c;for(let d=e;d<s;d+=e){let f=i[d],m=i[d+1];f<o&&(o=f),m<c&&(c=m),f>u&&(u=f),m>h&&(h=m)}l=Math.max(u-o,h-c),l=l!==0?32767/l:0}return Sr(r,a,e,o,c,l,0),a}function Nu(i,t,e,n,s){let r;if(s===up(i,t,e,n)>0)for(let a=t;a<e;a+=n)r=Uh(a/n|0,i[a],i[a+1],r);else for(let a=e-n;a>=t;a-=n)r=Uh(a/n|0,i[a],i[a+1],r);return r&&Cs(r,r.next)&&(Tr(r),r=r.next),r}function Gi(i,t){if(!i)return i;t||(t=i);let e=i,n;do if(n=!1,!e.steiner&&(Cs(e,e.next)||Te(e.prev,e,e.next)===0)){if(Tr(e),e=t=e.prev,e===e.next)break;n=!0}else e=e.next;while(n||e!==t);return t}function Sr(i,t,e,n,s,r,a){if(!i)return;!a&&r&&rp(i,n,s,r);let o=i;for(;i.prev!==i.next;){let c=i.prev,l=i.next;if(r?Kf(i,n,s,r):Jf(i)){t.push(c.i,i.i,l.i),Tr(i),i=l.next,o=l.next;continue}if(i=l,i===o){a?a===1?(i=jf(Gi(i),t),Sr(i,t,e,n,s,r,2)):a===2&&Qf(i,t,e,n,s,r):Sr(Gi(i),t,e,n,s,r,1);break}}}function Jf(i){let t=i.prev,e=i,n=i.next;if(Te(t,e,n)>=0)return!1;let s=t.x,r=e.x,a=n.x,o=t.y,c=e.y,l=n.y,u=Math.min(s,r,a),h=Math.min(o,c,l),d=Math.max(s,r,a),f=Math.max(o,c,l),m=n.next;for(;m!==t;){if(m.x>=u&&m.x<=d&&m.y>=h&&m.y<=f&&Js(s,o,r,c,a,l,m.x,m.y)&&Te(m.prev,m,m.next)>=0)return!1;m=m.next}return!0}function Kf(i,t,e,n){let s=i.prev,r=i,a=i.next;if(Te(s,r,a)>=0)return!1;let o=s.x,c=r.x,l=a.x,u=s.y,h=r.y,d=a.y,f=Math.min(o,c,l),m=Math.min(u,h,d),y=Math.max(o,c,l),g=Math.max(u,h,d),p=kl(f,m,t,e,n),v=kl(y,g,t,e,n),b=i.prevZ,_=i.nextZ;for(;b&&b.z>=p&&_&&_.z<=v;){if(b.x>=f&&b.x<=y&&b.y>=m&&b.y<=g&&b!==s&&b!==a&&Js(o,u,c,h,l,d,b.x,b.y)&&Te(b.prev,b,b.next)>=0||(b=b.prevZ,_.x>=f&&_.x<=y&&_.y>=m&&_.y<=g&&_!==s&&_!==a&&Js(o,u,c,h,l,d,_.x,_.y)&&Te(_.prev,_,_.next)>=0))return!1;_=_.nextZ}for(;b&&b.z>=p;){if(b.x>=f&&b.x<=y&&b.y>=m&&b.y<=g&&b!==s&&b!==a&&Js(o,u,c,h,l,d,b.x,b.y)&&Te(b.prev,b,b.next)>=0)return!1;b=b.prevZ}for(;_&&_.z<=v;){if(_.x>=f&&_.x<=y&&_.y>=m&&_.y<=g&&_!==s&&_!==a&&Js(o,u,c,h,l,d,_.x,_.y)&&Te(_.prev,_,_.next)>=0)return!1;_=_.nextZ}return!0}function jf(i,t){let e=i;do{let n=e.prev,s=e.next.next;!Cs(n,s)&&Fu(n,e,e.next,s)&&wr(n,s)&&wr(s,n)&&(t.push(n.i,e.i,s.i),Tr(e),Tr(e.next),e=i=s),e=e.next}while(e!==i);return Gi(e)}function Qf(i,t,e,n,s,r){let a=i;do{let o=a.next.next;for(;o!==a.prev;){if(a.i!==o.i&&lp(a,o)){let c=Ou(a,o);a=Gi(a,a.next),c=Gi(c,c.next),Sr(a,t,e,n,s,r,0),Sr(c,t,e,n,s,r,0);return}o=o.next}a=a.next}while(a!==i)}function tp(i,t,e,n){let s=[];for(let r=0,a=t.length;r<a;r++){let o=t[r]*n,c=r<a-1?t[r+1]*n:i.length,l=Nu(i,o,c,n,!1);l===l.next&&(l.steiner=!0),s.push(op(l))}s.sort(ep);for(let r=0;r<s.length;r++)e=np(s[r],e);return e}function ep(i,t){let e=i.x-t.x;if(e===0&&(e=i.y-t.y,e===0)){let n=(i.next.y-i.y)/(i.next.x-i.x),s=(t.next.y-t.y)/(t.next.x-t.x);e=n-s}return e}function np(i,t){let e=ip(i,t);if(!e)return t;let n=Ou(e,i);return Gi(n,n.next),Gi(e,e.next)}function ip(i,t){let e=t,n=i.x,s=i.y,r=-1/0,a;if(Cs(i,e))return e;do{if(Cs(i,e.next))return e.next;if(s<=e.y&&s>=e.next.y&&e.next.y!==e.y){let h=e.x+(s-e.y)*(e.next.x-e.x)/(e.next.y-e.y);if(h<=n&&h>r&&(r=h,a=e.x<e.next.x?e:e.next,h===n))return a}e=e.next}while(e!==t);if(!a)return null;let o=a,c=a.x,l=a.y,u=1/0;e=a;do{if(n>=e.x&&e.x>=c&&n!==e.x&&Uu(s<l?n:r,s,c,l,s<l?r:n,s,e.x,e.y)){let h=Math.abs(s-e.y)/(n-e.x);wr(e,i)&&(h<u||h===u&&(e.x>a.x||e.x===a.x&&sp(a,e)))&&(a=e,u=h)}e=e.next}while(e!==o);return a}function sp(i,t){return Te(i.prev,i,t.prev)<0&&Te(t.next,i,i.next)<0}function rp(i,t,e,n){let s=i;do s.z===0&&(s.z=kl(s.x,s.y,t,e,n)),s.prevZ=s.prev,s.nextZ=s.next,s=s.next;while(s!==i);s.prevZ.nextZ=null,s.prevZ=null,ap(s)}function ap(i){let t,e=1;do{let n=i,s;i=null;let r=null;for(t=0;n;){t++;let a=n,o=0;for(let l=0;l<e&&(o++,a=a.nextZ,!!a);l++);let c=e;for(;o>0||c>0&&a;)o!==0&&(c===0||!a||n.z<=a.z)?(s=n,n=n.nextZ,o--):(s=a,a=a.nextZ,c--),r?r.nextZ=s:i=s,s.prevZ=r,r=s;n=a}r.nextZ=null,e*=2}while(t>1);return i}function kl(i,t,e,n,s){return i=(i-e)*s|0,t=(t-n)*s|0,i=(i|i<<8)&16711935,i=(i|i<<4)&252645135,i=(i|i<<2)&858993459,i=(i|i<<1)&1431655765,t=(t|t<<8)&16711935,t=(t|t<<4)&252645135,t=(t|t<<2)&858993459,t=(t|t<<1)&1431655765,i|t<<1}function op(i){let t=i,e=i;do(t.x<e.x||t.x===e.x&&t.y<e.y)&&(e=t),t=t.next;while(t!==i);return e}function Uu(i,t,e,n,s,r,a,o){return(s-a)*(t-o)>=(i-a)*(r-o)&&(i-a)*(n-o)>=(e-a)*(t-o)&&(e-a)*(r-o)>=(s-a)*(n-o)}function Js(i,t,e,n,s,r,a,o){return!(i===a&&t===o)&&Uu(i,t,e,n,s,r,a,o)}function lp(i,t){return i.next.i!==t.i&&i.prev.i!==t.i&&!cp(i,t)&&(wr(i,t)&&wr(t,i)&&hp(i,t)&&(Te(i.prev,i,t.prev)||Te(i,t.prev,t))||Cs(i,t)&&Te(i.prev,i,i.next)>0&&Te(t.prev,t,t.next)>0)}function Te(i,t,e){return(t.y-i.y)*(e.x-t.x)-(t.x-i.x)*(e.y-t.y)}function Cs(i,t){return i.x===t.x&&i.y===t.y}function Fu(i,t,e,n){let s=Sa(Te(i,t,e)),r=Sa(Te(i,t,n)),a=Sa(Te(e,n,i)),o=Sa(Te(e,n,t));return!!(s!==r&&a!==o||s===0&&Ma(i,e,t)||r===0&&Ma(i,n,t)||a===0&&Ma(e,i,n)||o===0&&Ma(e,t,n))}function Ma(i,t,e){return t.x<=Math.max(i.x,e.x)&&t.x>=Math.min(i.x,e.x)&&t.y<=Math.max(i.y,e.y)&&t.y>=Math.min(i.y,e.y)}function Sa(i){return i>0?1:i<0?-1:0}function cp(i,t){let e=i;do{if(e.i!==i.i&&e.next.i!==i.i&&e.i!==t.i&&e.next.i!==t.i&&Fu(e,e.next,i,t))return!0;e=e.next}while(e!==i);return!1}function wr(i,t){return Te(i.prev,i,i.next)<0?Te(i,t,i.next)>=0&&Te(i,i.prev,t)>=0:Te(i,t,i.prev)<0||Te(i,i.next,t)<0}function hp(i,t){let e=i,n=!1,s=(i.x+t.x)/2,r=(i.y+t.y)/2;do e.y>r!=e.next.y>r&&e.next.y!==e.y&&s<(e.next.x-e.x)*(r-e.y)/(e.next.y-e.y)+e.x&&(n=!n),e=e.next;while(e!==i);return n}function Ou(i,t){let e=zl(i.i,i.x,i.y),n=zl(t.i,t.x,t.y),s=i.next,r=t.prev;return i.next=t,t.prev=i,e.next=s,s.prev=e,n.next=e,e.prev=n,r.next=n,n.prev=r,n}function Uh(i,t,e,n){let s=zl(i,t,e);return n?(s.next=n.next,s.prev=n,n.next.prev=s,n.next=s):(s.prev=s,s.next=s),s}function Tr(i){i.next.prev=i.prev,i.prev.next=i.next,i.prevZ&&(i.prevZ.nextZ=i.nextZ),i.nextZ&&(i.nextZ.prevZ=i.prevZ)}function zl(i,t,e){return{i,x:t,y:e,prev:null,next:null,z:0,prevZ:null,nextZ:null,steiner:!1}}function up(i,t,e,n){let s=0;for(let r=t,a=e-n;r<e;r+=n)s+=(i[a]-i[r])*(i[r+1]+i[a+1]),a=r;return s}var Gl=class{static triangulate(t,e,n=2){return Zf(t,e,n)}},Un=class i{static area(t){let e=t.length,n=0;for(let s=e-1,r=0;r<e;s=r++)n+=t[s].x*t[r].y-t[r].x*t[s].y;return n*.5}static isClockWise(t){return i.area(t)<0}static triangulateShape(t,e){let n=[],s=[],r=[];Fh(t),Oh(n,t);let a=t.length;e.forEach(Fh);for(let c=0;c<e.length;c++)s.push(a),a+=e[c].length,Oh(n,e[c]);let o=Gl.triangulate(n,s);for(let c=0;c<o.length;c+=3)r.push(o.slice(c,c+3));return r}};function Fh(i){let t=i.length;t>2&&i[t-1].equals(i[0])&&i.pop()}function Oh(i,t){for(let e=0;e<t.length;e++)i.push(t[e].x),i.push(t[e].y)}var Vi=class i extends de{constructor(t=new zi([new ft(.5,.5),new ft(-.5,.5),new ft(-.5,-.5),new ft(.5,-.5)]),e={}){super(),this.type="ExtrudeGeometry",this.parameters={shapes:t,options:e},t=Array.isArray(t)?t:[t];let n=this,s=[],r=[];for(let o=0,c=t.length;o<c;o++){let l=t[o];a(l)}this.setAttribute("position",new Jt(s,3)),this.setAttribute("uv",new Jt(r,2)),this.computeVertexNormals();function a(o){let c=[],l=e.curveSegments!==void 0?e.curveSegments:12,u=e.steps!==void 0?e.steps:1,h=e.depth!==void 0?e.depth:1,d=e.bevelEnabled!==void 0?e.bevelEnabled:!0,f=e.bevelThickness!==void 0?e.bevelThickness:.2,m=e.bevelSize!==void 0?e.bevelSize:f-.1,y=e.bevelOffset!==void 0?e.bevelOffset:0,g=e.bevelSegments!==void 0?e.bevelSegments:3,p=e.extrudePath,v=e.UVGenerator!==void 0?e.UVGenerator:dp,b,_=!1,T,w,x,M;if(p){b=p.getSpacedPoints(u),_=!0,d=!1;let E=p.isCatmullRomCurve3?p.closed:!1;T=p.computeFrenetFrames(u,E),w=new F,x=new F,M=new F}d||(g=0,f=0,m=0,y=0);let P=o.extractPoints(l),C=P.shape,k=P.holes;if(!Un.isClockWise(C)){C=C.reverse();for(let E=0,$=k.length;E<$;E++){let B=k[E];Un.isClockWise(B)&&(k[E]=B.reverse())}}function j(E){let B=10000000000000001e-36,nt=E[0];for(let it=1;it<=E.length;it++){let pt=it%E.length,mt=E[pt],X=mt.x-nt.x,Z=mt.y-nt.y,bt=X*X+Z*Z,H=Math.max(Math.abs(mt.x),Math.abs(mt.y),Math.abs(nt.x),Math.abs(nt.y)),Qt=B*H*H;if(bt<=Qt){E.splice(pt,1),it--;continue}nt=mt}}j(C),k.forEach(j);let D=k.length,K=C;for(let E=0;E<D;E++){let $=k[E];C=C.concat($)}function Q(E,$,B){return $||Zt("ExtrudeGeometry: vec does not exist"),E.clone().addScaledVector($,B)}let Y=C.length;function J(E,$,B){let nt,it,pt,mt=E.x-$.x,X=E.y-$.y,Z=B.x-E.x,bt=B.y-E.y,H=mt*mt+X*X,Qt=mt*bt-X*Z;if(Math.abs(Qt)>Number.EPSILON){let jt=Math.sqrt(H),se=Math.sqrt(Z*Z+bt*bt),Mt=$.x-X/jt,fe=$.y+mt/jt,Gt=B.x-bt/se,$t=B.y+Z/se,R=((Gt-Mt)*bt-($t-fe)*Z)/(mt*bt-X*Z);nt=Mt+mt*R-E.x,it=fe+X*R-E.y;let S=nt*nt+it*it;if(S<=2)return new ft(nt,it);pt=Math.sqrt(S/2)}else{let jt=!1;mt>Number.EPSILON?Z>Number.EPSILON&&(jt=!0):mt<-Number.EPSILON?Z<-Number.EPSILON&&(jt=!0):Math.sign(X)===Math.sign(bt)&&(jt=!0),jt?(nt=-X,it=mt,pt=Math.sqrt(H)):(nt=mt,it=X,pt=Math.sqrt(H/2))}return new ft(nt/pt,it/pt)}let at=[];for(let E=0,$=K.length,B=$-1,nt=E+1;E<$;E++,B++,nt++)B===$&&(B=0),nt===$&&(nt=0),at[E]=J(K[E],K[B],K[nt]);let W=[],tt,vt=at.concat();for(let E=0,$=D;E<$;E++){let B=k[E];tt=[];for(let nt=0,it=B.length,pt=it-1,mt=nt+1;nt<it;nt++,pt++,mt++)pt===it&&(pt=0),mt===it&&(mt=0),tt[nt]=J(B[nt],B[pt],B[mt]);W.push(tt),vt=vt.concat(tt)}let wt;if(g===0)wt=Un.triangulateShape(K,k);else{let E=[],$=[];for(let B=0;B<g;B++){let nt=B/g,it=f*Math.cos(nt*Math.PI/2),pt=m*Math.sin(nt*Math.PI/2)+y;for(let mt=0,X=K.length;mt<X;mt++){let Z=Q(K[mt],at[mt],pt);L(Z.x,Z.y,-it),nt===0&&E.push(Z)}for(let mt=0,X=D;mt<X;mt++){let Z=k[mt];tt=W[mt];let bt=[];for(let H=0,Qt=Z.length;H<Qt;H++){let jt=Q(Z[H],tt[H],pt);L(jt.x,jt.y,-it),nt===0&&bt.push(jt)}nt===0&&$.push(bt)}}wt=Un.triangulateShape(E,$)}let st=wt.length,U=m+y;for(let E=0;E<Y;E++){let $=d?Q(C[E],vt[E],U):C[E];_?(x.copy(T.normals[0]).multiplyScalar($.x),w.copy(T.binormals[0]).multiplyScalar($.y),M.copy(b[0]).add(x).add(w),L(M.x,M.y,M.z)):L($.x,$.y,0)}for(let E=1;E<=u;E++)for(let $=0;$<Y;$++){let B=d?Q(C[$],vt[$],U):C[$];_?(x.copy(T.normals[E]).multiplyScalar(B.x),w.copy(T.binormals[E]).multiplyScalar(B.y),M.copy(b[E]).add(x).add(w),L(M.x,M.y,M.z)):L(B.x,B.y,h/u*E)}for(let E=g-1;E>=0;E--){let $=E/g,B=f*Math.cos($*Math.PI/2),nt=m*Math.sin($*Math.PI/2)+y;for(let it=0,pt=K.length;it<pt;it++){let mt=Q(K[it],at[it],nt);L(mt.x,mt.y,h+B)}for(let it=0,pt=k.length;it<pt;it++){let mt=k[it];tt=W[it];for(let X=0,Z=mt.length;X<Z;X++){let bt=Q(mt[X],tt[X],nt);_?L(bt.x,bt.y+b[u-1].y,b[u-1].x+B):L(bt.x,bt.y,h+B)}}}et(),V();function et(){let E=s.length/3;if(d){let $=0,B=Y*$;for(let nt=0;nt<st;nt++){let it=wt[nt];I(it[2]+B,it[1]+B,it[0]+B)}$=u+g*2,B=Y*$;for(let nt=0;nt<st;nt++){let it=wt[nt];I(it[0]+B,it[1]+B,it[2]+B)}}else{for(let $=0;$<st;$++){let B=wt[$];I(B[2],B[1],B[0])}for(let $=0;$<st;$++){let B=wt[$];I(B[0]+Y*u,B[1]+Y*u,B[2]+Y*u)}}n.addGroup(E,s.length/3-E,0)}function V(){let E=s.length/3,$=0;N(K,$),$+=K.length;for(let B=0,nt=k.length;B<nt;B++){let it=k[B];N(it,$),$+=it.length}n.addGroup(E,s.length/3-E,1)}function N(E,$){let B=E.length;for(;--B>=0;){let nt=B,it=B-1;it<0&&(it=E.length-1);for(let pt=0,mt=u+g*2;pt<mt;pt++){let X=Y*pt,Z=Y*(pt+1),bt=$+nt+X,H=$+it+X,Qt=$+it+Z,jt=$+nt+Z;G(bt,H,Qt,jt)}}}function L(E,$,B){c.push(E),c.push($),c.push(B)}function I(E,$,B){ht(E),ht($),ht(B);let nt=s.length/3,it=v.generateTopUV(n,s,nt-3,nt-2,nt-1);ut(it[0]),ut(it[1]),ut(it[2])}function G(E,$,B,nt){ht(E),ht($),ht(nt),ht($),ht(B),ht(nt);let it=s.length/3,pt=v.generateSideWallUV(n,s,it-6,it-3,it-2,it-1);ut(pt[0]),ut(pt[1]),ut(pt[3]),ut(pt[1]),ut(pt[2]),ut(pt[3])}function ht(E){s.push(c[E*3+0]),s.push(c[E*3+1]),s.push(c[E*3+2])}function ut(E){r.push(E.x),r.push(E.y)}}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}toJSON(){let t=super.toJSON(),e=this.parameters.shapes,n=this.parameters.options;return fp(e,n,t)}static fromJSON(t,e){let n=[];for(let r=0,a=t.shapes.length;r<a;r++){let o=e[t.shapes[r]];n.push(o)}let s=t.options.extrudePath;return s!==void 0&&(t.options.extrudePath=new Xa[s.type]().fromJSON(s)),new i(n,t.options)}},dp={generateTopUV:function(i,t,e,n,s){let r=t[e*3],a=t[e*3+1],o=t[n*3],c=t[n*3+1],l=t[s*3],u=t[s*3+1];return[new ft(r,a),new ft(o,c),new ft(l,u)]},generateSideWallUV:function(i,t,e,n,s,r){let a=t[e*3],o=t[e*3+1],c=t[e*3+2],l=t[n*3],u=t[n*3+1],h=t[n*3+2],d=t[s*3],f=t[s*3+1],m=t[s*3+2],y=t[r*3],g=t[r*3+1],p=t[r*3+2];return Math.abs(o-u)<Math.abs(a-l)?[new ft(a,1-c),new ft(l,1-h),new ft(d,1-m),new ft(y,1-p)]:[new ft(o,1-c),new ft(u,1-h),new ft(f,1-m),new ft(g,1-p)]}};function fp(i,t,e){if(e.shapes=[],Array.isArray(i))for(let n=0,s=i.length;n<s;n++){let r=i[n];e.shapes.push(r.uuid)}else e.shapes.push(i.uuid);return e.options=Object.assign({},t),t.extrudePath!==void 0&&(e.options.extrudePath=t.extrudePath.toJSON()),e}var Ar=class i extends gr{constructor(t=1,e=0){let n=(1+Math.sqrt(5))/2,s=[-1,n,0,1,n,0,-1,-n,0,1,-n,0,0,-1,n,0,1,n,0,-1,-n,0,1,-n,n,0,-1,n,0,1,-n,0,-1,-n,0,1],r=[0,11,5,0,5,1,0,1,7,0,7,10,0,10,11,1,5,9,5,11,4,11,10,2,10,7,6,7,1,8,3,9,4,3,4,2,3,2,6,3,6,8,3,8,9,4,9,5,2,4,11,6,2,10,8,6,7,9,8,1];super(s,r,t,e),this.type="IcosahedronGeometry",this.parameters={radius:t,detail:e}}static fromJSON(t){return new i(t.radius,t.detail)}};var ni=class i extends de{constructor(t=1,e=1,n=1,s=1){super(),this.type="PlaneGeometry",this.parameters={width:t,height:e,widthSegments:n,heightSegments:s};let r=t/2,a=e/2,o=Math.floor(n),c=Math.floor(s),l=o+1,u=c+1,h=t/o,d=e/c,f=[],m=[],y=[],g=[];for(let p=0;p<u;p++){let v=p*d-a;for(let b=0;b<l;b++){let _=b*h-r;m.push(_,-v,0),y.push(0,0,1),g.push(b/o),g.push(1-p/c)}}for(let p=0;p<c;p++)for(let v=0;v<o;v++){let b=v+l*p,_=v+l*(p+1),T=v+1+l*(p+1),w=v+1+l*p;f.push(b,_,w),f.push(_,T,w)}this.setIndex(f),this.setAttribute("position",new Jt(m,3)),this.setAttribute("normal",new Jt(y,3)),this.setAttribute("uv",new Jt(g,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new i(t.width,t.height,t.widthSegments,t.heightSegments)}},Er=class i extends de{constructor(t=.5,e=1,n=32,s=1,r=0,a=Math.PI*2){super(),this.type="RingGeometry",this.parameters={innerRadius:t,outerRadius:e,thetaSegments:n,phiSegments:s,thetaStart:r,thetaLength:a},n=Math.max(3,n),s=Math.max(1,s);let o=[],c=[],l=[],u=[],h=t,d=(e-t)/s,f=new F,m=new ft;for(let y=0;y<=s;y++){for(let g=0;g<=n;g++){let p=r+g/n*a;f.x=h*Math.cos(p),f.y=h*Math.sin(p),c.push(f.x,f.y,f.z),l.push(0,0,1),m.x=(f.x/e+1)/2,m.y=(f.y/e+1)/2,u.push(m.x,m.y)}h+=d}for(let y=0;y<s;y++){let g=y*(n+1);for(let p=0;p<n;p++){let v=p+g,b=v,_=v+n+1,T=v+n+2,w=v+1;o.push(b,_,w),o.push(_,T,w)}}this.setIndex(o),this.setAttribute("position",new Jt(c,3)),this.setAttribute("normal",new Jt(l,3)),this.setAttribute("uv",new Jt(u,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new i(t.innerRadius,t.outerRadius,t.thetaSegments,t.phiSegments,t.thetaStart,t.thetaLength)}};var De=class i extends de{constructor(t=1,e=32,n=16,s=0,r=Math.PI*2,a=0,o=Math.PI){super(),this.type="SphereGeometry",this.parameters={radius:t,widthSegments:e,heightSegments:n,phiStart:s,phiLength:r,thetaStart:a,thetaLength:o},e=Math.max(3,Math.floor(e)),n=Math.max(2,Math.floor(n));let c=Math.min(a+o,Math.PI),l=0,u=[],h=new F,d=new F,f=[],m=[],y=[],g=[];for(let p=0;p<=n;p++){let v=[],b=p/n,_=a+b*o,T=t*Math.cos(_),w=Math.sqrt(t*t-T*T),x=0;p===0&&a===0?x=.5/e:p===n&&c===Math.PI&&(x=-.5/e);for(let M=0;M<=e;M++){let P=M/e,C=s+P*r;h.x=-w*Math.cos(C),h.y=T,h.z=w*Math.sin(C),m.push(h.x,h.y,h.z),d.copy(h).normalize(),y.push(d.x,d.y,d.z),g.push(P+x,1-b),v.push(l++)}u.push(v)}for(let p=0;p<n;p++)for(let v=0;v<e;v++){let b=u[p][v+1],_=u[p][v],T=u[p+1][v],w=u[p+1][v+1];(p!==0||a>0)&&f.push(b,_,w),(p!==n-1||c<Math.PI)&&f.push(_,T,w)}this.setIndex(f),this.setAttribute("position",new Jt(m,3)),this.setAttribute("normal",new Jt(y,3)),this.setAttribute("uv",new Jt(g,2))}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}static fromJSON(t){return new i(t.radius,t.widthSegments,t.heightSegments,t.phiStart,t.phiLength,t.thetaStart,t.thetaLength)}};var un=class i extends de{constructor(t=new br(new F(-1,-1,0),new F(-1,1,0),new F(1,1,0)),e=64,n=1,s=8,r=!1){super(),this.type="TubeGeometry",this.parameters={path:t,tubularSegments:e,radius:n,radialSegments:s,closed:r};let a=t.computeFrenetFrames(e,r);this.tangents=a.tangents,this.normals=a.normals,this.binormals=a.binormals;let o=new F,c=new F,l=new ft,u=new F,h=[],d=[],f=[],m=[];y(),this.setIndex(m),this.setAttribute("position",new Jt(h,3)),this.setAttribute("normal",new Jt(d,3)),this.setAttribute("uv",new Jt(f,2));function y(){for(let b=0;b<e;b++)g(b);g(r===!1?e:0),v(),p()}function g(b){u=t.getPointAt(b/e,u);let _=a.normals[b],T=a.binormals[b];for(let w=0;w<=s;w++){let x=w/s*Math.PI*2,M=Math.sin(x),P=-Math.cos(x);c.x=P*_.x+M*T.x,c.y=P*_.y+M*T.y,c.z=P*_.z+M*T.z,c.normalize(),d.push(c.x,c.y,c.z),o.x=u.x+n*c.x,o.y=u.y+n*c.y,o.z=u.z+n*c.z,h.push(o.x,o.y,o.z)}}function p(){for(let b=1;b<=e;b++)for(let _=1;_<=s;_++){let T=(s+1)*(b-1)+(_-1),w=(s+1)*b+(_-1),x=(s+1)*b+_,M=(s+1)*(b-1)+_;m.push(T,w,M),m.push(w,x,M)}}function v(){for(let b=0;b<=e;b++)for(let _=0;_<=s;_++)l.x=b/e,l.y=_/s,f.push(l.x,l.y)}}copy(t){return super.copy(t),this.parameters=Object.assign({},t.parameters),this}toJSON(){let t=super.toJSON();return t.path=this.parameters.path.toJSON(),t}static fromJSON(t){return new i(new Xa[t.path.type]().fromJSON(t.path),t.tubularSegments,t.radius,t.radialSegments,t.closed)}};function Ji(i){let t={};for(let e in i){t[e]={};for(let n in i[e]){let s=i[e][n];if(Bh(s))s.isRenderTargetTexture?(qt("UniformsUtils: Textures of render targets cannot be cloned via cloneUniforms() or mergeUniforms()."),t[e][n]=null):t[e][n]=s.clone();else if(Array.isArray(s))if(Bh(s[0])){let r=[];for(let a=0,o=s.length;a<o;a++)r[a]=s[a].clone();t[e][n]=r}else t[e][n]=s.slice();else t[e][n]=s}}return t}function Ze(i){let t={};for(let e=0;e<i.length;e++){let n=Ji(i[e]);for(let s in n)t[s]=n[s]}return t}function Bh(i){return i&&(i.isColor||i.isMatrix3||i.isMatrix4||i.isVector2||i.isVector3||i.isVector4||i.isTexture||i.isQuaternion)}function pp(i){let t=[];for(let e=0;e<i.length;e++)t.push(i[e].clone());return t}function wc(i){let t=i.getRenderTarget();return t===null?i.outputColorSpace:t.isXRRenderTarget===!0?t.texture.colorSpace:oe.workingColorSpace}var Bu={clone:Ji,merge:Ze},mp=`void main() {
	gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}`,gp=`void main() {
	gl_FragColor = vec4( 1.0, 0.0, 0.0, 1.0 );
}`,dn=class extends ti{constructor(t){super(),this.isShaderMaterial=!0,this.type="ShaderMaterial",this.defines={},this.uniforms={},this.uniformsGroups=[],this.vertexShader=mp,this.fragmentShader=gp,this.linewidth=1,this.wireframe=!1,this.wireframeLinewidth=1,this.fog=!1,this.lights=!1,this.clipping=!1,this.forceSinglePass=!0,this.extensions={clipCullDistance:!1,multiDraw:!1},this.defaultAttributeValues={color:[1,1,1],uv:[0,0],uv1:[0,0]},this.index0AttributeName=void 0,this.uniformsNeedUpdate=!1,this.glslVersion=null,t!==void 0&&this.setValues(t)}copy(t){return super.copy(t),this.fragmentShader=t.fragmentShader,this.vertexShader=t.vertexShader,this.uniforms=Ji(t.uniforms),this.uniformsGroups=pp(t.uniformsGroups),this.defines=Object.assign({},t.defines),this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.fog=t.fog,this.lights=t.lights,this.clipping=t.clipping,this.extensions=Object.assign({},t.extensions),this.glslVersion=t.glslVersion,this.defaultAttributeValues=Object.assign({},t.defaultAttributeValues),this.index0AttributeName=t.index0AttributeName,this.uniformsNeedUpdate=t.uniformsNeedUpdate,this}toJSON(t){let e=super.toJSON(t);e.glslVersion=this.glslVersion,e.uniforms={};for(let s in this.uniforms){let a=this.uniforms[s].value;a&&a.isTexture?e.uniforms[s]={type:"t",value:a.toJSON(t).uuid}:a&&a.isColor?e.uniforms[s]={type:"c",value:a.getHex()}:a&&a.isVector2?e.uniforms[s]={type:"v2",value:a.toArray()}:a&&a.isVector3?e.uniforms[s]={type:"v3",value:a.toArray()}:a&&a.isVector4?e.uniforms[s]={type:"v4",value:a.toArray()}:a&&a.isMatrix3?e.uniforms[s]={type:"m3",value:a.toArray()}:a&&a.isMatrix4?e.uniforms[s]={type:"m4",value:a.toArray()}:e.uniforms[s]={value:a}}Object.keys(this.defines).length>0&&(e.defines=this.defines),e.vertexShader=this.vertexShader,e.fragmentShader=this.fragmentShader,e.lights=this.lights,e.clipping=this.clipping;let n={};for(let s in this.extensions)this.extensions[s]===!0&&(n[s]=!0);return Object.keys(n).length>0&&(e.extensions=n),e}fromJSON(t,e){if(super.fromJSON(t,e),t.uniforms!==void 0)for(let n in t.uniforms){let s=t.uniforms[n];switch(this.uniforms[n]={},s.type){case"t":this.uniforms[n].value=e[s.value]||null;break;case"c":this.uniforms[n].value=new Vt().setHex(s.value);break;case"v2":this.uniforms[n].value=new ft().fromArray(s.value);break;case"v3":this.uniforms[n].value=new F().fromArray(s.value);break;case"v4":this.uniforms[n].value=new Se().fromArray(s.value);break;case"m3":this.uniforms[n].value=new Xt().fromArray(s.value);break;case"m4":this.uniforms[n].value=new xe().fromArray(s.value);break;default:this.uniforms[n].value=s.value}}if(t.defines!==void 0&&(this.defines=t.defines),t.vertexShader!==void 0&&(this.vertexShader=t.vertexShader),t.fragmentShader!==void 0&&(this.fragmentShader=t.fragmentShader),t.glslVersion!==void 0&&(this.glslVersion=t.glslVersion),t.extensions!==void 0)for(let n in t.extensions)this.extensions[n]=t.extensions[n];return t.lights!==void 0&&(this.lights=t.lights),t.clipping!==void 0&&(this.clipping=t.clipping),this}},qa=class extends dn{constructor(t){super(t),this.isRawShaderMaterial=!0,this.type="RawShaderMaterial"}},Re=class extends ti{constructor(t){super(),this.isMeshStandardMaterial=!0,this.type="MeshStandardMaterial",this.defines={STANDARD:""},this.color=new Vt(16777215),this.roughness=1,this.metalness=0,this.map=null,this.lightMap=null,this.lightMapIntensity=1,this.aoMap=null,this.aoMapIntensity=1,this.emissive=new Vt(0),this.emissiveIntensity=1,this.emissiveMap=null,this.bumpMap=null,this.bumpScale=1,this.normalMap=null,this.normalMapType=Yo,this.normalScale=new ft(1,1),this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.roughnessMap=null,this.metalnessMap=null,this.alphaMap=null,this.envMap=null,this.envMapRotation=new jn,this.envMapIntensity=1,this.wireframe=!1,this.wireframeLinewidth=1,this.wireframeLinecap="round",this.wireframeLinejoin="round",this.flatShading=!1,this.fog=!0,this.setValues(t)}copy(t){return super.copy(t),this.defines={STANDARD:""},this.color.copy(t.color),this.roughness=t.roughness,this.metalness=t.metalness,this.map=t.map,this.lightMap=t.lightMap,this.lightMapIntensity=t.lightMapIntensity,this.aoMap=t.aoMap,this.aoMapIntensity=t.aoMapIntensity,this.emissive.copy(t.emissive),this.emissiveMap=t.emissiveMap,this.emissiveIntensity=t.emissiveIntensity,this.bumpMap=t.bumpMap,this.bumpScale=t.bumpScale,this.normalMap=t.normalMap,this.normalMapType=t.normalMapType,this.normalScale.copy(t.normalScale),this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.roughnessMap=t.roughnessMap,this.metalnessMap=t.metalnessMap,this.alphaMap=t.alphaMap,this.envMap=t.envMap,this.envMapRotation.copy(t.envMapRotation),this.envMapIntensity=t.envMapIntensity,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this.wireframeLinecap=t.wireframeLinecap,this.wireframeLinejoin=t.wireframeLinejoin,this.flatShading=t.flatShading,this.fog=t.fog,this}};var Ya=class extends ti{constructor(t){super(),this.isMeshDepthMaterial=!0,this.type="MeshDepthMaterial",this.depthPacking=_u,this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.wireframe=!1,this.wireframeLinewidth=1,this.setValues(t)}copy(t){return super.copy(t),this.depthPacking=t.depthPacking,this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this.wireframe=t.wireframe,this.wireframeLinewidth=t.wireframeLinewidth,this}},$a=class extends ti{constructor(t){super(),this.isMeshDistanceMaterial=!0,this.type="MeshDistanceMaterial",this.map=null,this.alphaMap=null,this.displacementMap=null,this.displacementScale=1,this.displacementBias=0,this.setValues(t)}copy(t){return super.copy(t),this.map=t.map,this.alphaMap=t.alphaMap,this.displacementMap=t.displacementMap,this.displacementScale=t.displacementScale,this.displacementBias=t.displacementBias,this}};var Hi=class extends Fi{constructor(t){super(),this.isLineDashedMaterial=!0,this.type="LineDashedMaterial",this.scale=1,this.dashSize=3,this.gapSize=1,this.setValues(t)}copy(t){return super.copy(t),this.scale=t.scale,this.dashSize=t.dashSize,this.gapSize=t.gapSize,this}};function hs(i,t){return!i||i.constructor===t?i:typeof t.BYTES_PER_ELEMENT=="number"?new t(i):Array.prototype.slice.call(i)}function wa(i){return i!==void 0&&i.inTangents!==void 0&&i.outTangents!==void 0}var yi=class{constructor(t,e,n,s){this.parameterPositions=t,this._cachedIndex=0,this.resultBuffer=s!==void 0?s:new e.constructor(n),this.sampleValues=e,this.valueSize=n,this.settings=null,this.DefaultSettings_={}}evaluate(t){let e=this.parameterPositions,n=this._cachedIndex,s=e[n],r=e[n-1];n:{t:{let a;e:{i:if(!(t<s)){for(let o=n+2;;){if(s===void 0){if(t<r)break i;return n=e.length,this._cachedIndex=n,this.copySampleValue_(n-1)}if(n===o)break;if(r=s,s=e[++n],t<s)break t}a=e.length;break e}if(!(t>=r)){let o=e[1];t<o&&(n=2,r=o);for(let c=n-2;;){if(r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(n===c)break;if(s=r,r=e[--n-1],t>=r)break t}a=n,n=0;break e}break n}for(;n<a;){let o=n+a>>>1;t<e[o]?a=o:n=o+1}if(s=e[n],r=e[n-1],r===void 0)return this._cachedIndex=0,this.copySampleValue_(0);if(s===void 0)return n=e.length,this._cachedIndex=n,this.copySampleValue_(n-1)}this._cachedIndex=n,this.intervalChanged_(n,r,s)}return this.interpolate_(n,r,t,s)}getSettings_(){return this.settings||this.DefaultSettings_}copySampleValue_(t){let e=this.resultBuffer,n=this.sampleValues,s=this.valueSize,r=t*s;for(let a=0;a!==s;++a)e[a]=n[r+a];return e}interpolate_(){throw new Error("THREE.Interpolant: Call to abstract method.")}intervalChanged_(){}},Za=class extends yi{constructor(t,e,n,s){super(t,e,n,s),this._weightPrev=-0,this._offsetPrev=-0,this._weightNext=-0,this._offsetNext=-0,this.DefaultSettings_={endingStart:Fl,endingEnd:Fl}}intervalChanged_(t,e,n){let s=this.parameterPositions,r=t-2,a=t+1,o=s[r],c=s[a];if(o===void 0)switch(this.getSettings_().endingStart){case Ol:r=t,o=2*e-n;break;case Bl:r=s.length-2,o=e+s[r]-s[r+1];break;default:r=t,o=n}if(c===void 0)switch(this.getSettings_().endingEnd){case Ol:a=t,c=2*n-e;break;case Bl:a=1,c=n+s[1]-s[0];break;default:a=t-1,c=e}let l=(n-e)*.5,u=this.valueSize;this._weightPrev=l/(e-o),this._weightNext=l/(c-n),this._offsetPrev=r*u,this._offsetNext=a*u}interpolate_(t,e,n,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=t*o,l=c-o,u=this._offsetPrev,h=this._offsetNext,d=this._weightPrev,f=this._weightNext,m=(n-e)/(s-e),y=m*m,g=y*m,p=-d*g+2*d*y-d*m,v=(1+d)*g+(-1.5-2*d)*y+(-.5+d)*m+1,b=(-1-f)*g+(1.5+f)*y+.5*m,_=f*g-f*y;for(let T=0;T!==o;++T)r[T]=p*a[u+T]+v*a[l+T]+b*a[c+T]+_*a[h+T];return r}},Ja=class extends yi{constructor(t,e,n,s){super(t,e,n,s)}interpolate_(t,e,n,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=t*o,l=c-o,u=(n-e)/(s-e),h=1-u;for(let d=0;d!==o;++d)r[d]=a[l+d]*h+a[c+d]*u;return r}},Ka=class extends yi{constructor(t,e,n,s){super(t,e,n,s)}interpolate_(t){return this.copySampleValue_(t-1)}},ja=class extends yi{interpolate_(t,e,n,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=t*o,l=c-o,u=this.inTangents,h=this.outTangents;if(!u||!h){let m=(n-e)/(s-e),y=1-m;for(let g=0;g!==o;++g)r[g]=a[l+g]*y+a[c+g]*m;return r}let d=o*2,f=t-1;for(let m=0;m!==o;++m){let y=a[l+m],g=a[c+m],p=f*d+m*2,v=h[p],b=h[p+1],_=t*d+m*2,T=u[_],w=u[_+1],x=_p(n,e,v,T,s);r[m]=ku(x,y,b,w,g)}return r}};function ku(i,t,e,n,s){let r=1-i;return r*r*r*t+3*r*r*i*e+3*r*i*i*n+i*i*i*s}function xp(i,t,e,n,s){let r=1-i;return 3*r*r*(e-t)+6*r*i*(n-e)+3*i*i*(s-n)}function _p(i,t,e,n,s){let r=(i-t)/(s-t);for(let a=0;a<8;a++){let o=ku(r,t,e,n,s)-i;if(Math.abs(o)<1e-10)break;let c=xp(r,t,e,n,s);if(Math.abs(c)<1e-10)break;r=Math.max(0,Math.min(1,r-o/c))}return r}var fn=class{constructor(t,e,n,s){if(t===void 0)throw new Error("THREE.KeyframeTrack: track name is undefined");if(e===void 0||e.length===0)throw new Error("THREE.KeyframeTrack: no keyframes in track named "+t);this.name=t,this.times=hs(e,this.TimeBufferType),this.values=hs(n,this.ValueBufferType),this.setInterpolation(s||this.DefaultInterpolation)}static toJSON(t){let e=t.constructor,n;if(e.toJSON!==this.toJSON)n=e.toJSON(t);else{n={name:t.name,times:hs(t.times,Array),values:hs(t.values,Array)};let s=t.getInterpolation();s!==t.DefaultInterpolation&&(n.interpolation=s),wa(t.settings)&&(n.settings={inTangents:hs(t.settings.inTangents,Array),outTangents:hs(t.settings.outTangents,Array)})}return n.type=t.ValueTypeName,n}InterpolantFactoryMethodDiscrete(t){return new Ka(this.times,this.values,this.getValueSize(),t)}InterpolantFactoryMethodLinear(t){return new Ja(this.times,this.values,this.getValueSize(),t)}InterpolantFactoryMethodSmooth(t){return new Za(this.times,this.values,this.getValueSize(),t)}InterpolantFactoryMethodBezier(t){let e=new ja(this.times,this.values,this.getValueSize(),t);return this.settings&&(e.inTangents=this.settings.inTangents,e.outTangents=this.settings.outTangents),e}setInterpolation(t){let e;switch(t){case nr:e=this.InterpolantFactoryMethodDiscrete;break;case Fa:e=this.InterpolantFactoryMethodLinear;break;case Ea:e=this.InterpolantFactoryMethodSmooth;break;case Ul:e=this.InterpolantFactoryMethodBezier;break}if(e===void 0){let n="unsupported interpolation for "+this.ValueTypeName+" keyframe track named "+this.name;if(this.createInterpolant===void 0)if(t!==this.DefaultInterpolation)this.setInterpolation(this.DefaultInterpolation);else throw new Error(n);return qt("KeyframeTrack:",n),this}return this.createInterpolant=e,this}getInterpolation(){switch(this.createInterpolant){case this.InterpolantFactoryMethodDiscrete:return nr;case this.InterpolantFactoryMethodLinear:return Fa;case this.InterpolantFactoryMethodSmooth:return Ea;case this.InterpolantFactoryMethodBezier:return Ul}}getValueSize(){return this.values.length/this.times.length}shift(t){if(t!==0){let e=this.times;for(let n=0,s=e.length;n!==s;++n)e[n]+=t;wa(this.settings)&&(kh(this.settings.inTangents,t),kh(this.settings.outTangents,t))}return this}scale(t){if(t!==1){let e=this.times;for(let n=0,s=e.length;n!==s;++n)e[n]*=t;wa(this.settings)&&(zh(this.settings.inTangents,t),zh(this.settings.outTangents,t))}return this}trim(t,e){let n=this.times,s=n.length,r=0,a=s-1;for(;r!==s&&n[r]<t;)++r;for(;a!==-1&&n[a]>e;)--a;if(++a,r!==0||a!==s){r>=a&&(a=Math.max(a,1),r=a-1);let o=this.getValueSize();this.times=n.slice(r,a),this.values=this.values.slice(r*o,a*o)}return this}validate(){let t=!0,e=this.getValueSize();e-Math.floor(e)!==0&&(Zt("KeyframeTrack: Invalid value size in track.",this),t=!1);let n=this.times,s=this.values,r=n.length;r===0&&(Zt("KeyframeTrack: Track is empty.",this),t=!1);let a=null;for(let o=0;o!==r;o++){let c=n[o];if(typeof c=="number"&&isNaN(c)){Zt("KeyframeTrack: Time is not a valid number.",this,o,c),t=!1;break}if(a!==null&&a>c){Zt("KeyframeTrack: Out of order keys.",this,o,c,a),t=!1;break}a=c}if(s!==void 0&&of(s))for(let o=0,c=s.length;o!==c;++o){let l=s[o];if(isNaN(l)){Zt("KeyframeTrack: Value is not a valid number.",this,o,l),t=!1;break}}return t}optimize(){let t=this.times.slice(),e=this.values.slice(),n=this.getValueSize(),s=this.getInterpolation()===Ea,r=t.length-1,a=1;for(let o=1;o<r;++o){let c=!1,l=t[o],u=t[o+1];if(l!==u&&(o!==1||l!==t[0]))if(s)c=!0;else{let h=o*n,d=h-n,f=h+n;for(let m=0;m!==n;++m){let y=e[h+m];if(y!==e[d+m]||y!==e[f+m]){c=!0;break}}}if(c){if(o!==a){t[a]=t[o];let h=o*n,d=a*n;for(let f=0;f!==n;++f)e[d+f]=e[h+f]}++a}}if(r>0){t[a]=t[r];for(let o=r*n,c=a*n,l=0;l!==n;++l)e[c+l]=e[o+l];++a}return a!==t.length?(this.times=t.slice(0,a),this.values=e.slice(0,a*n)):(this.times=t,this.values=e),this}clone(){let t=this.times.slice(),e=this.values.slice(),n=this.constructor,s=new n(this.name,t,e);return s.createInterpolant=this.createInterpolant,wa(this.settings)&&(s.settings={inTangents:this.settings.inTangents.slice(),outTangents:this.settings.outTangents.slice()}),s}};function kh(i,t){for(let e=0,n=i.length;e!==n;e+=2)i[e]+=t}function zh(i,t){for(let e=0,n=i.length;e!==n;e+=2)i[e]*=t}fn.prototype.ValueTypeName="";fn.prototype.TimeBufferType=Float32Array;fn.prototype.ValueBufferType=Float32Array;fn.prototype.DefaultInterpolation=Fa;var vi=class extends fn{constructor(t,e,n){super(t,e,n)}};vi.prototype.ValueTypeName="bool";vi.prototype.ValueBufferType=Array;vi.prototype.DefaultInterpolation=nr;vi.prototype.InterpolantFactoryMethodLinear=void 0;vi.prototype.InterpolantFactoryMethodSmooth=void 0;var Qa=class extends fn{constructor(t,e,n,s){super(t,e,n,s)}};Qa.prototype.ValueTypeName="color";var to=class extends fn{constructor(t,e,n,s){super(t,e,n,s)}};to.prototype.ValueTypeName="number";var eo=class extends yi{constructor(t,e,n,s){super(t,e,n,s)}interpolate_(t,e,n,s){let r=this.resultBuffer,a=this.sampleValues,o=this.valueSize,c=(n-e)/(s-e),l=t*o;for(let u=l+o;l!==u;l+=4)ln.slerpFlat(r,0,a,l-o,a,l,c);return r}},Cr=class extends fn{constructor(t,e,n,s){super(t,e,n,s)}InterpolantFactoryMethodLinear(t){return new eo(this.times,this.values,this.getValueSize(),t)}};Cr.prototype.ValueTypeName="quaternion";Cr.prototype.InterpolantFactoryMethodSmooth=void 0;var bi=class extends fn{constructor(t,e,n){super(t,e,n)}};bi.prototype.ValueTypeName="string";bi.prototype.ValueBufferType=Array;bi.prototype.DefaultInterpolation=nr;bi.prototype.InterpolantFactoryMethodLinear=void 0;bi.prototype.InterpolantFactoryMethodSmooth=void 0;var no=class extends fn{constructor(t,e,n,s){super(t,e,n,s)}};no.prototype.ValueTypeName="vector";var Vl={enabled:!1,files:{},add:function(i,t){this.enabled!==!1&&(Gh(i)||(this.files[i]=t))},get:function(i){if(this.enabled!==!1&&!Gh(i))return this.files[i]},remove:function(i){delete this.files[i]},clear:function(){this.files={}}};function Gh(i){try{let t=i.slice(i.indexOf(":")+1);return new URL(t).protocol==="blob:"}catch{return!1}}var io=class{constructor(t,e,n){let s=this,r=!1,a=0,o=0,c,l=[];this.onStart=void 0,this.onLoad=t,this.onProgress=e,this.onError=n,this._abortController=null,this.itemStart=function(u){o++,r===!1&&s.onStart!==void 0&&s.onStart(u,a,o),r=!0},this.itemEnd=function(u){a++,s.onProgress!==void 0&&s.onProgress(u,a,o),a===o&&(r=!1,s.onLoad!==void 0&&s.onLoad())},this.itemError=function(u){s.onError!==void 0&&s.onError(u)},this.resolveURL=function(u){return u=u.normalize("NFC"),c?c(u):u},this.setURLModifier=function(u){return c=u,this},this.addHandler=function(u,h){return l.push(u,h),this},this.removeHandler=function(u){let h=l.indexOf(u);return h!==-1&&l.splice(h,2),this},this.getHandler=function(u){for(let h=0,d=l.length;h<d;h+=2){let f=l[h],m=l[h+1];if(f.global&&(f.lastIndex=0),f.test(u))return m}return null},this.abort=function(){return this.abortController.abort(),this._abortController=null,this}}get abortController(){return this._abortController||(this._abortController=new AbortController),this._abortController}},zu=new io,Wi=class{constructor(t){this.manager=t!==void 0?t:zu,this.crossOrigin="anonymous",this.withCredentials=!1,this.path="",this.resourcePath="",this.requestHeader={},this.priority="auto",this.cache="default",typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}load(){}loadAsync(t,e){let n=this;return new Promise(function(s,r){n.load(t,s,e,r)})}parse(){}setCrossOrigin(t){return this.crossOrigin=t,this}setWithCredentials(t){return this.withCredentials=t,this}setPath(t){return this.path=t,this}setResourcePath(t){return this.resourcePath=t,this}setRequestHeader(t){return this.requestHeader=t,this}setPriority(t){return this.priority=t,this}setCache(t){return this.cache=t,this}abort(){return this}};Wi.DEFAULT_MATERIAL_NAME="__DEFAULT";var Zn={},Hl=class extends Error{constructor(t,e){super(t),this.response=e}},Rr=class extends Wi{constructor(t){super(t),this.mimeType="",this.responseType="",this._abortController=new AbortController}load(t,e,n,s){t===void 0&&(t=""),this.path!==void 0&&(t=this.path+t),t=this.manager.resolveURL(t);let r=Vl.get(`file:${t}`);if(r!==void 0){this.manager.itemStart(t),setTimeout(()=>{e&&e(r),this.manager.itemEnd(t)},0);return}if(Zn[t]!==void 0){Zn[t].push({onLoad:e,onProgress:n,onError:s});return}Zn[t]=[],Zn[t].push({onLoad:e,onProgress:n,onError:s});let a=new Request(t,{headers:new Headers(this.requestHeader),credentials:this.withCredentials?"include":"same-origin",priority:this.priority,cache:this.cache,signal:typeof AbortSignal.any=="function"?AbortSignal.any([this._abortController.signal,this.manager.abortController.signal]):this._abortController.signal}),o=this.mimeType,c=this.responseType;fetch(a).then(l=>{if(l.status===200||l.status===0){if(l.status===0&&qt("FileLoader: HTTP Status 0 received."),typeof ReadableStream>"u"||l.body===void 0||l.body.getReader===void 0)return l;let u=Zn[t],h=l.body.getReader(),d=l.headers.get("X-File-Size")||l.headers.get("Content-Length"),f=d?parseInt(d):0,m=f!==0,y=0,g=new ReadableStream({start(p){v();function v(){h.read().then(({done:b,value:_})=>{if(b)p.close();else{y+=_.byteLength;let T=new ProgressEvent("progress",{lengthComputable:m,loaded:y,total:f});for(let w=0,x=u.length;w<x;w++){let M=u[w];M.onProgress&&M.onProgress(T)}p.enqueue(_),v()}},b=>{p.error(b)})}}});return new Response(g)}else throw new Hl(`fetch for "${l.url}" responded with ${l.status}: ${l.statusText}`,l)}).then(l=>{switch(c){case"arraybuffer":return l.arrayBuffer();case"blob":return l.blob();case"document":return l.text().then(u=>new DOMParser().parseFromString(u,o));case"json":return l.json();default:if(o==="")return l.text();{let h=/charset="?([^;"\s]*)"?/i.exec(o),d=h&&h[1]?h[1].toLowerCase():void 0,f=new TextDecoder(d);return l.arrayBuffer().then(m=>f.decode(m))}}}).then(l=>{Vl.add(`file:${t}`,l);let u=Zn[t];delete Zn[t];for(let h=0,d=u.length;h<d;h++){let f=u[h];f.onLoad&&f.onLoad(l)}}).catch(l=>{let u=Zn[t];if(u===void 0)throw this.manager.itemError(t),l;delete Zn[t];for(let h=0,d=u.length;h<d;h++){let f=u[h];f.onError&&f.onError(l)}this.manager.itemError(t)}).finally(()=>{this.manager.itemEnd(t)}),this.manager.itemStart(t)}setResponseType(t){return this.responseType=t,this}setMimeType(t){return this.mimeType=t,this}abort(){return this._abortController.abort(),this._abortController=new AbortController,this}};var Pr=class extends Ne{static registerNode(t){this.prototype._lightNode=t}constructor(t,e=1){super(),this.isLight=!0,this.type="Light",this.color=new Vt(t),this.intensity=e}copy(t,e){return super.copy(t,e),this.color.copy(t.color),this.intensity=t.intensity,this}toJSON(t){let e=super.toJSON(t);return e.object.color=this.color.getHex(),e.object.intensity=this.intensity,e}},An=class extends Pr{constructor(t,e,n){super(t,n),this.isHemisphereLight=!0,this.type="HemisphereLight",this.position.copy(Ne.DEFAULT_UP),this.updateMatrix(),this.groundColor=new Vt(e)}copy(t,e){return super.copy(t,e),this.groundColor.copy(t.groundColor),this}toJSON(t){let e=super.toJSON(t);return e.object.groundColor=this.groundColor.getHex(),e}},Nl=new xe,Vh=new F,Hh=new F,so=class{constructor(t){this.camera=t,this.intensity=1,this.bias=0,this.biasNode=null,this.normalBias=0,this.radius=1,this.blurSamples=8,this.mapSize=new ft(512,512),this.mapType=nn,this.map=null,this.mapPass=null,this.matrix=new xe,this.autoUpdate=!0,this.needsUpdate=!1,this._frustum=new _i,this._frameExtents=new ft(1,1),this._viewportCount=1,this._viewports=[new Se(0,0,1,1)]}getViewportCount(){return this._viewportCount}getCamera(){return this.camera}getFrustum(){return this._frustum}updateMatrices(t){let e=this.camera;Vh.setFromMatrixPosition(t.matrixWorld),e.position.copy(Vh),Hh.setFromMatrixPosition(t.target.matrixWorld),e.lookAt(Hh),e.updateMatrixWorld(),this._updateMatrix(e,this.matrix,this._frustum)}_updateMatrix(t,e,n,s){Nl.multiplyMatrices(t.projectionMatrix,t.matrixWorldInverse),n.setFromProjectionMatrix(Nl,t.coordinateSystem,t.reversedDepth);let r=this._frameExtents,a=s?s.z/r.x:1,o=s?s.w/r.y:1,c=s?s.x/r.x:0,l=s?s.y/r.y:0;t.coordinateSystem===xs||t.reversedDepth?e.set(.5*a,0,0,.5*a+c,0,.5*o,0,.5*o+l,0,0,1,0,0,0,0,1):e.set(.5*a,0,0,.5*a+c,0,.5*o,0,.5*o+l,0,0,.5,.5,0,0,0,1),e.multiply(Nl)}getViewport(t){return this._viewports[t]}getFrameExtents(){return this._frameExtents}dispose(){this.map&&this.map.dispose(),this.mapPass&&this.mapPass.dispose()}copy(t){return this.camera=t.camera.clone(),this.intensity=t.intensity,this.bias=t.bias,this.radius=t.radius,this.autoUpdate=t.autoUpdate,this.needsUpdate=t.needsUpdate,this.normalBias=t.normalBias,this.blurSamples=t.blurSamples,this.mapSize.copy(t.mapSize),this.mapType=t.mapType,this.biasNode=t.biasNode,this}clone(){return new this.constructor().copy(this)}toJSON(){let t={};return t.intensity=this.intensity,t.bias=this.bias,t.normalBias=this.normalBias,t.radius=this.radius,t.blurSamples=this.blurSamples,t.mapSize=this.mapSize.toArray(),t.mapType=this.mapType,t.camera=this.camera.toJSON(!1).object,delete t.camera.matrix,t}},Ta=new F,Aa=new ln,Nn=new F,Ir=class extends Ne{constructor(){super(),this.isCamera=!0,this.type="Camera",this.matrixWorldInverse=new xe,this.projectionMatrix=new xe,this.projectionMatrixInverse=new xe,this.coordinateSystem=gn,this._reversedDepth=!1}get reversedDepth(){return this._reversedDepth}copy(t,e){return super.copy(t,e),this.matrixWorldInverse.copy(t.matrixWorldInverse),this.projectionMatrix.copy(t.projectionMatrix),this.projectionMatrixInverse.copy(t.projectionMatrixInverse),this.coordinateSystem=t.coordinateSystem,this}getWorldDirection(t){return super.getWorldDirection(t).negate()}updateMatrixWorld(t){super.updateMatrixWorld(t),this.matrixWorld.decompose(Ta,Aa,Nn),Nn.x===1&&Nn.y===1&&Nn.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(Ta,Aa,Nn.set(1,1,1)).invert()}updateWorldMatrix(t,e,n=!1){super.updateWorldMatrix(t,e,n),this.matrixWorld.decompose(Ta,Aa,Nn),Nn.x===1&&Nn.y===1&&Nn.z===1?this.matrixWorldInverse.copy(this.matrixWorld).invert():this.matrixWorldInverse.compose(Ta,Aa,Nn.set(1,1,1)).invert()}clone(){return new this.constructor().copy(this)}},mi=new F,Wh=new ft,Xh=new ft,Ke=class extends Ir{constructor(t=50,e=1,n=.1,s=2e3){super(),this.isPerspectiveCamera=!0,this.type="PerspectiveCamera",this.fov=t,this.zoom=1,this.near=n,this.far=s,this.focus=10,this.aspect=e,this.view=null,this.filmGauge=35,this.filmOffset=0,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.fov=t.fov,this.zoom=t.zoom,this.near=t.near,this.far=t.far,this.focus=t.focus,this.aspect=t.aspect,this.view=t.view===null?null:Object.assign({},t.view),this.filmGauge=t.filmGauge,this.filmOffset=t.filmOffset,this}setFocalLength(t){let e=.5*this.getFilmHeight()/t;this.fov=ys*2*Math.atan(e),this.updateProjectionMatrix()}getFocalLength(){let t=Math.tan(js*.5*this.fov);return .5*this.getFilmHeight()/t}getEffectiveFOV(){return ys*2*Math.atan(Math.tan(js*.5*this.fov)/this.zoom)}getFilmWidth(){return this.filmGauge*Math.min(this.aspect,1)}getFilmHeight(){return this.filmGauge/Math.max(this.aspect,1)}getViewBounds(t,e,n){mi.set(-1,-1,.5).applyMatrix4(this.projectionMatrixInverse),e.set(mi.x,mi.y).multiplyScalar(-t/mi.z),mi.set(1,1,.5).applyMatrix4(this.projectionMatrixInverse),n.set(mi.x,mi.y).multiplyScalar(-t/mi.z)}getViewSize(t,e){return this.getViewBounds(t,Wh,Xh),e.subVectors(Xh,Wh)}setViewOffset(t,e,n,s,r,a){this.aspect=t/e,this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let t=this.near,e=t*Math.tan(js*.5*this.fov)/this.zoom,n=2*e,s=this.aspect*n,r=-.5*s,a=this.view;if(this.view!==null&&this.view.enabled){let c=a.fullWidth,l=a.fullHeight;r+=a.offsetX*s/c,e-=a.offsetY*n/l,s*=a.width/c,n*=a.height/l}let o=this.filmOffset;o!==0&&(r+=t*o/this.getFilmWidth()),this.projectionMatrix.makePerspective(r,r+s,e,e-n,t,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){let e=super.toJSON(t);return e.object.fov=this.fov,e.object.zoom=this.zoom,e.object.near=this.near,e.object.far=this.far,e.object.focus=this.focus,e.object.aspect=this.aspect,this.view!==null&&(e.object.view=Object.assign({},this.view)),e.object.filmGauge=this.filmGauge,e.object.filmOffset=this.filmOffset,e}};var En=class extends Ir{constructor(t=-1,e=1,n=1,s=-1,r=.1,a=2e3){super(),this.isOrthographicCamera=!0,this.type="OrthographicCamera",this.zoom=1,this.view=null,this.left=t,this.right=e,this.top=n,this.bottom=s,this.near=r,this.far=a,this.updateProjectionMatrix()}copy(t,e){return super.copy(t,e),this.left=t.left,this.right=t.right,this.top=t.top,this.bottom=t.bottom,this.near=t.near,this.far=t.far,this.zoom=t.zoom,this.view=t.view===null?null:Object.assign({},t.view),this}setViewOffset(t,e,n,s,r,a){this.view===null&&(this.view={enabled:!0,fullWidth:1,fullHeight:1,offsetX:0,offsetY:0,width:1,height:1}),this.view.enabled=!0,this.view.fullWidth=t,this.view.fullHeight=e,this.view.offsetX=n,this.view.offsetY=s,this.view.width=r,this.view.height=a,this.updateProjectionMatrix()}clearViewOffset(){this.view!==null&&(this.view.enabled=!1),this.updateProjectionMatrix()}updateProjectionMatrix(){let t=(this.right-this.left)/(2*this.zoom),e=(this.top-this.bottom)/(2*this.zoom),n=(this.right+this.left)/2,s=(this.top+this.bottom)/2,r=n-t,a=n+t,o=s+e,c=s-e;if(this.view!==null&&this.view.enabled){let l=(this.right-this.left)/this.view.fullWidth/this.zoom,u=(this.top-this.bottom)/this.view.fullHeight/this.zoom;r+=l*this.view.offsetX,a=r+l*this.view.width,o-=u*this.view.offsetY,c=o-u*this.view.height}this.projectionMatrix.makeOrthographic(r,a,o,c,this.near,this.far,this.coordinateSystem,this.reversedDepth),this.projectionMatrixInverse.copy(this.projectionMatrix).invert()}toJSON(t){let e=super.toJSON(t);return e.object.zoom=this.zoom,e.object.left=this.left,e.object.right=this.right,e.object.top=this.top,e.object.bottom=this.bottom,e.object.near=this.near,e.object.far=this.far,this.view!==null&&(e.object.view=Object.assign({},this.view)),e}},Wl=class extends so{constructor(){super(new En(-5,5,5,-5,.5,500)),this.isDirectionalLightShadow=!0}},Cn=class extends Pr{constructor(t,e){super(t,e),this.isDirectionalLight=!0,this.type="DirectionalLight",this.position.copy(Ne.DEFAULT_UP),this.updateMatrix(),this.target=new Ne,this.shadow=new Wl}dispose(){super.dispose(),this.shadow.dispose()}copy(t){return super.copy(t),this.target=t.target.clone(),this.shadow=t.shadow.clone(),this}toJSON(t){let e=super.toJSON(t);return e.object.shadow=this.shadow.toJSON(),e.object.target=this.target.uuid,e}};var us=-90,ds=1,Rs=class extends Ne{constructor(t,e,n){super(),this.type="CubeCamera",this.renderTarget=n,this.coordinateSystem=null,this.activeMipmapLevel=0;let s=new Ke(us,ds,t,e);s.layers=this.layers,this.add(s);let r=new Ke(us,ds,t,e);r.layers=this.layers,this.add(r);let a=new Ke(us,ds,t,e);a.layers=this.layers,this.add(a);let o=new Ke(us,ds,t,e);o.layers=this.layers,this.add(o);let c=new Ke(us,ds,t,e);c.layers=this.layers,this.add(c);let l=new Ke(us,ds,t,e);l.layers=this.layers,this.add(l)}updateCoordinateSystem(){let t=this.coordinateSystem,e=this.children.concat(),[n,s,r,a,o,c]=e;for(let l of e)this.remove(l);if(t===gn)n.up.set(0,1,0),n.lookAt(1,0,0),s.up.set(0,1,0),s.lookAt(-1,0,0),r.up.set(0,0,-1),r.lookAt(0,1,0),a.up.set(0,0,1),a.lookAt(0,-1,0),o.up.set(0,1,0),o.lookAt(0,0,1),c.up.set(0,1,0),c.lookAt(0,0,-1);else if(t===xs)n.up.set(0,-1,0),n.lookAt(-1,0,0),s.up.set(0,-1,0),s.lookAt(1,0,0),r.up.set(0,0,1),r.lookAt(0,1,0),a.up.set(0,0,-1),a.lookAt(0,-1,0),o.up.set(0,-1,0),o.lookAt(0,0,1),c.up.set(0,-1,0),c.lookAt(0,0,-1);else throw new Error("THREE.CubeCamera.updateCoordinateSystem(): Invalid coordinate system: "+t);for(let l of e)this.add(l),l.updateMatrixWorld()}update(t,e){this.parent===null&&this.updateMatrixWorld();let{renderTarget:n,activeMipmapLevel:s}=this;this.coordinateSystem!==t.coordinateSystem&&(this.coordinateSystem=t.coordinateSystem,this.updateCoordinateSystem());let[r,a,o,c,l,u]=this.children,h=t.getRenderTarget(),d=t.getActiveCubeFace(),f=t.getActiveMipmapLevel(),m=t.xr.enabled;t.xr.enabled=!1;let y=n.texture.mipmapsAutoUpdate;n.texture.mipmapsAutoUpdate=!1;let g=!1;t.isWebGLRenderer===!0?g=t.state.buffers.depth.getReversed():g=t.reversedDepthBuffer,t.setRenderTarget(n,0,s),g&&t.autoClear===!1&&t.clearDepth(),t.render(e,r),t.setRenderTarget(n,1,s),g&&t.autoClear===!1&&t.clearDepth(),t.render(e,a),t.setRenderTarget(n,2,s),g&&t.autoClear===!1&&t.clearDepth(),t.render(e,o),t.setRenderTarget(n,3,s),g&&t.autoClear===!1&&t.clearDepth(),t.render(e,c),t.setRenderTarget(n,4,s),g&&t.autoClear===!1&&t.clearDepth(),t.render(e,l),n.texture.mipmapsAutoUpdate=y,t.setRenderTarget(n,5,s),g&&t.autoClear===!1&&t.clearDepth(),t.render(e,u),t.setRenderTarget(h,d,f),t.xr.enabled=m,n.texture.needsPMREMUpdate=!0}},ro=class extends Ke{constructor(t=[]){super(),this.isArrayCamera=!0,this.isMultiViewCamera=!1,this.cameras=t,this.perCameraFrustumCulled=!1}};var Tc="\\[\\]\\.:\\/",yp=new RegExp("["+Tc+"]","g"),Ac="[^"+Tc+"]",vp="[^"+Tc.replace("\\.","")+"]",bp=/((?:WC+[\/:])*)/.source.replace("WC",Ac),Mp=/(WCOD+)?/.source.replace("WCOD",vp),Sp=/(?:\.(WC+)(?:\[(.+)\])?)?/.source.replace("WC",Ac),wp=/\.(WC+)(?:\[(.+)\])?/.source.replace("WC",Ac),Tp=new RegExp("^"+bp+Mp+Sp+wp+"$"),Ap=["material","materials","bones","map"],Xl=class{constructor(t,e,n){let s=n||Me.parseTrackName(e);this._targetGroup=t,this._bindings=t.subscribe_(e,s)}getValue(t,e){this.bind();let n=this._targetGroup.nCachedObjects_,s=this._bindings[n];s!==void 0&&s.getValue(t,e)}setValue(t,e){let n=this._bindings;for(let s=this._targetGroup.nCachedObjects_,r=n.length;s!==r;++s)n[s].setValue(t,e)}bind(){let t=this._bindings;for(let e=this._targetGroup.nCachedObjects_,n=t.length;e!==n;++e)t[e].bind()}unbind(){let t=this._bindings;for(let e=this._targetGroup.nCachedObjects_,n=t.length;e!==n;++e)t[e].unbind()}},Me=class i{constructor(t,e,n){this.path=e,this.parsedPath=n||i.parseTrackName(e),this.node=i.findNode(t,this.parsedPath.nodeName),this.rootNode=t,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}static create(t,e,n){return t&&t.isAnimationObjectGroup?new i.Composite(t,e,n):new i(t,e,n)}static sanitizeNodeName(t){return t.replace(/\s/g,"_").replace(yp,"")}static parseTrackName(t){let e=Tp.exec(t);if(e===null)throw new Error("THREE.PropertyBinding: Cannot parse trackName: "+t);let n={nodeName:e[2],objectName:e[3],objectIndex:e[4],propertyName:e[5],propertyIndex:e[6]},s=n.nodeName&&n.nodeName.lastIndexOf(".");if(s!==void 0&&s!==-1){let r=n.nodeName.substring(s+1);Ap.indexOf(r)!==-1&&(n.nodeName=n.nodeName.substring(0,s),n.objectName=r)}if(n.propertyName===null||n.propertyName.length===0)throw new Error("THREE.PropertyBinding: can not parse propertyName from trackName: "+t);return n}static findNode(t,e){if(e===void 0||e===""||e==="."||e===-1||e===t.name||e===t.uuid)return t;if(t.skeleton){let n=t.skeleton.getBoneByName(e);if(n!==void 0)return n}if(t.children){let n=function(r){for(let a=0;a<r.length;a++){let o=r[a];if(o.name===e||o.uuid===e)return o;let c=n(o.children);if(c)return c}return null},s=n(t.children);if(s)return s}return null}_getValue_unavailable(){}_setValue_unavailable(){}_getValue_direct(t,e){t[e]=this.targetObject[this.propertyName]}_getValue_array(t,e){let n=this.resolvedProperty;for(let s=0,r=n.length;s!==r;++s)t[e++]=n[s]}_getValue_arrayElement(t,e){t[e]=this.resolvedProperty[this.propertyIndex]}_getValue_toArray(t,e){this.resolvedProperty.toArray(t,e)}_setValue_direct(t,e){this.targetObject[this.propertyName]=t[e]}_setValue_direct_setNeedsUpdate(t,e){this.targetObject[this.propertyName]=t[e],this.targetObject.needsUpdate=!0}_setValue_direct_setMatrixWorldNeedsUpdate(t,e){this.targetObject[this.propertyName]=t[e],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_array(t,e){let n=this.resolvedProperty;for(let s=0,r=n.length;s!==r;++s)n[s]=t[e++]}_setValue_array_setNeedsUpdate(t,e){let n=this.resolvedProperty;for(let s=0,r=n.length;s!==r;++s)n[s]=t[e++];this.targetObject.needsUpdate=!0}_setValue_array_setMatrixWorldNeedsUpdate(t,e){let n=this.resolvedProperty;for(let s=0,r=n.length;s!==r;++s)n[s]=t[e++];this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_arrayElement(t,e){this.resolvedProperty[this.propertyIndex]=t[e]}_setValue_arrayElement_setNeedsUpdate(t,e){this.resolvedProperty[this.propertyIndex]=t[e],this.targetObject.needsUpdate=!0}_setValue_arrayElement_setMatrixWorldNeedsUpdate(t,e){this.resolvedProperty[this.propertyIndex]=t[e],this.targetObject.matrixWorldNeedsUpdate=!0}_setValue_fromArray(t,e){this.resolvedProperty.fromArray(t,e)}_setValue_fromArray_setNeedsUpdate(t,e){this.resolvedProperty.fromArray(t,e),this.targetObject.needsUpdate=!0}_setValue_fromArray_setMatrixWorldNeedsUpdate(t,e){this.resolvedProperty.fromArray(t,e),this.targetObject.matrixWorldNeedsUpdate=!0}_getValue_unbound(t,e){this.bind(),this.getValue(t,e)}_setValue_unbound(t,e){this.bind(),this.setValue(t,e)}bind(){let t=this.node,e=this.parsedPath,n=e.objectName,s=e.propertyName,r=e.propertyIndex;if(t||(t=i.findNode(this.rootNode,e.nodeName),this.node=t),this.getValue=this._getValue_unavailable,this.setValue=this._setValue_unavailable,!t){qt("PropertyBinding: No target node found for track: "+this.path+".");return}if(n){let l=e.objectIndex;switch(n){case"materials":if(!t.material){Zt("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!t.material.materials){Zt("PropertyBinding: Can not bind to material.materials as node.material does not have a materials array.",this);return}t=t.material.materials;break;case"bones":if(!t.skeleton){Zt("PropertyBinding: Can not bind to bones as node does not have a skeleton.",this);return}t=t.skeleton.bones;for(let u=0;u<t.length;u++)if(t[u].name===l){l=u;break}break;case"map":if("map"in t){t=t.map;break}if(!t.material){Zt("PropertyBinding: Can not bind to material as node does not have a material.",this);return}if(!t.material.map){Zt("PropertyBinding: Can not bind to material.map as node.material does not have a map.",this);return}t=t.material.map;break;default:if(t[n]===void 0){Zt("PropertyBinding: Can not bind to objectName of node undefined.",this);return}t=t[n]}if(l!==void 0){if(t[l]===void 0){Zt("PropertyBinding: Trying to bind to objectIndex of objectName, but is undefined.",this,t);return}t=t[l]}}let a=t[s];if(a===void 0){let l=e.nodeName;Zt("PropertyBinding: Trying to update property for track: "+l+"."+s+" but it wasn't found.",t);return}let o=this.Versioning.None;this.targetObject=t,t.isMaterial===!0?o=this.Versioning.NeedsUpdate:t.isObject3D===!0&&(o=this.Versioning.MatrixWorldNeedsUpdate);let c=this.BindingType.Direct;if(r!==void 0){if(s==="morphTargetInfluences"){if(!t.geometry){Zt("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.",this);return}if(!t.geometry.morphAttributes){Zt("PropertyBinding: Can not bind to morphTargetInfluences because node does not have a geometry.morphAttributes.",this);return}t.morphTargetDictionary[r]!==void 0&&(r=t.morphTargetDictionary[r])}c=this.BindingType.ArrayElement,this.resolvedProperty=a,this.propertyIndex=r}else a.fromArray!==void 0&&a.toArray!==void 0?(c=this.BindingType.HasFromToArray,this.resolvedProperty=a):Array.isArray(a)?(c=this.BindingType.EntireArray,this.resolvedProperty=a):this.propertyName=s;this.getValue=this.GetterByBindingType[c],this.setValue=this.SetterByBindingTypeAndVersioning[c][o]}unbind(){this.node=null,this.getValue=this._getValue_unbound,this.setValue=this._setValue_unbound}};Me.Composite=Xl;Me.prototype.BindingType={Direct:0,EntireArray:1,ArrayElement:2,HasFromToArray:3};Me.prototype.Versioning={None:0,NeedsUpdate:1,MatrixWorldNeedsUpdate:2};Me.prototype.GetterByBindingType=[Me.prototype._getValue_direct,Me.prototype._getValue_array,Me.prototype._getValue_arrayElement,Me.prototype._getValue_toArray];Me.prototype.SetterByBindingTypeAndVersioning=[[Me.prototype._setValue_direct,Me.prototype._setValue_direct_setNeedsUpdate,Me.prototype._setValue_direct_setMatrixWorldNeedsUpdate],[Me.prototype._setValue_array,Me.prototype._setValue_array_setNeedsUpdate,Me.prototype._setValue_array_setMatrixWorldNeedsUpdate],[Me.prototype._setValue_arrayElement,Me.prototype._setValue_arrayElement_setNeedsUpdate,Me.prototype._setValue_arrayElement_setMatrixWorldNeedsUpdate],[Me.prototype._setValue_fromArray,Me.prototype._setValue_fromArray_setNeedsUpdate,Me.prototype._setValue_fromArray_setMatrixWorldNeedsUpdate]];var dy=new Float32Array(1);var qh=new xe,ii=class{constructor(t,e,n=0,s=1/0){this.ray=new xi(t,e),this.near=n,this.far=s,this.camera=null,this.layers=new bs,this.params={Mesh:{},Line:{threshold:1},LOD:{},Points:{threshold:1},Sprite:{}}}set(t,e){this.ray.set(t,e)}setFromCamera(t,e){e.isPerspectiveCamera?(this.ray.origin.setFromMatrixPosition(e.matrixWorld),this.ray.direction.set(t.x,t.y,.5).unproject(e).sub(this.ray.origin).normalize(),this.camera=e):e.isOrthographicCamera?(this.ray.origin.set(t.x,t.y,e.projectionMatrix.elements[14]).unproject(e),this.ray.direction.set(0,0,-1).transformDirection(e.matrixWorld),this.camera=e):Zt("Raycaster: Unsupported camera type: "+e.type)}setFromXRController(t){return qh.identity().extractRotation(t.matrixWorld),this.ray.origin.setFromMatrixPosition(t.matrixWorld),this.ray.direction.set(0,0,-1).applyMatrix4(qh),this}intersectObject(t,e=!0,n=[]){return ql(t,this,n,e),n.sort(Yh),n}intersectObjects(t,e=!0,n=[]){for(let s=0,r=t.length;s<r;s++)ql(t[s],this,n,e);return n.sort(Yh),n}};function Yh(i,t){return i.distance-t.distance}function ql(i,t,e,n){let s=!0;if(i.layers.test(t.layers)&&i.raycast(t,e)===!1&&(s=!1),s===!0&&n===!0){let r=i.children;for(let a=0,o=r.length;a<o;a++)ql(r[a],t,e,!0)}}var Ps=class{constructor(t=1,e=0,n=0){this.radius=t,this.phi=e,this.theta=n}set(t,e,n){return this.radius=t,this.phi=e,this.theta=n,this}copy(t){return this.radius=t.radius,this.phi=t.phi,this.theta=t.theta,this}makeSafe(){return this.phi=ne(this.phi,1e-6,Math.PI-1e-6),this}setFromVector3(t){return this.setFromCartesianCoords(t.x,t.y,t.z)}setFromCartesianCoords(t,e,n){return this.radius=Math.sqrt(t*t+e*e+n*n),this.radius===0?(this.theta=0,this.phi=0):(this.theta=Math.atan2(t,n),this.phi=Math.acos(ne(e/this.radius,-1,1))),this}clone(){return new this.constructor().copy(this)}};var Yl=class{static{this.prototype.isMatrix2=!0}constructor(t,e,n,s){this.elements=[1,0,0,1],t!==void 0&&this.set(t,e,n,s)}identity(){return this.set(1,0,0,1),this}fromArray(t,e=0){for(let n=0;n<4;n++)this.elements[n]=t[n+e];return this}set(t,e,n,s){let r=this.elements;return r[0]=t,r[2]=e,r[1]=n,r[3]=s,this}},$h=new ft,Is=class{constructor(t=new ft(1/0,1/0),e=new ft(-1/0,-1/0)){this.isBox2=!0,this.min=t,this.max=e}set(t,e){return this.min.copy(t),this.max.copy(e),this}setFromPoints(t){this.makeEmpty();for(let e=0,n=t.length;e<n;e++)this.expandByPoint(t[e]);return this}setFromCenterAndSize(t,e){let n=$h.copy(e).multiplyScalar(.5);return this.min.copy(t).sub(n),this.max.copy(t).add(n),this}clone(){return new this.constructor().copy(this)}copy(t){return this.min.copy(t.min),this.max.copy(t.max),this}makeEmpty(){return this.min.x=this.min.y=1/0,this.max.x=this.max.y=-1/0,this}isEmpty(){return this.max.x<this.min.x||this.max.y<this.min.y}getCenter(t){return this.isEmpty()?t.set(0,0):t.addVectors(this.min,this.max).multiplyScalar(.5)}getSize(t){return this.isEmpty()?t.set(0,0):t.subVectors(this.max,this.min)}expandByPoint(t){return this.min.min(t),this.max.max(t),this}expandByVector(t){return this.min.sub(t),this.max.add(t),this}expandByScalar(t){return this.min.addScalar(-t),this.max.addScalar(t),this}containsPoint(t){return t.x>=this.min.x&&t.x<=this.max.x&&t.y>=this.min.y&&t.y<=this.max.y}containsBox(t){return this.min.x<=t.min.x&&t.max.x<=this.max.x&&this.min.y<=t.min.y&&t.max.y<=this.max.y}getParameter(t,e){return e.set((t.x-this.min.x)/(this.max.x-this.min.x),(t.y-this.min.y)/(this.max.y-this.min.y))}intersectsBox(t){return t.max.x>=this.min.x&&t.min.x<=this.max.x&&t.max.y>=this.min.y&&t.min.y<=this.max.y}clampPoint(t,e){return e.copy(t).clamp(this.min,this.max)}distanceToPoint(t){return this.clampPoint(t,$h).distanceTo(t)}intersect(t){return this.min.max(t.min),this.max.min(t.max),this.isEmpty()&&this.makeEmpty(),this}union(t){return this.min.min(t.min),this.max.max(t.max),this}translate(t){return this.min.add(t),this.max.add(t),this}equals(t){return t.min.equals(this.min)&&t.max.equals(this.max)}};var Rn=class{constructor(){this.type="ShapePath",this.color=new Vt,this.subPaths=[],this.currentPath=null,this.userData={}}moveTo(t,e){return this.currentPath=new On,this.subPaths.push(this.currentPath),this.currentPath.moveTo(t,e),this}lineTo(t,e){return this.currentPath.lineTo(t,e),this}quadraticCurveTo(t,e,n,s){return this.currentPath.quadraticCurveTo(t,e,n,s),this}bezierCurveTo(t,e,n,s,r,a){return this.currentPath.bezierCurveTo(t,e,n,s,r,a),this}splineThru(t){return this.currentPath.splineThru(t),this}toShapes(){function t(c,l){let u=!1,h=l.length;for(let d=0,f=h-1;d<h;f=d++){let m=l[d],y=l[f];m.y>c.y!=y.y>c.y&&c.x<(y.x-m.x)*(c.y-m.y)/(y.y-m.y)+m.x&&(u=!u)}return u}function e(c,l){let u=l.getCenter(new ft);if(t(u,c))return u;let h=u.y,d=[],f=c.length;for(let m=0;m<f;m++){let y=c[m],g=c[(m+1)%f];if(y.y>h!=g.y>h){let p=y.x+(h-y.y)*(g.x-y.x)/(g.y-y.y);d.push(p)}}return d.length>1&&(d.sort((m,y)=>m-y),u.x=(d[0]+d[1])/2),u}let n=this.userData.style&&this.userData.style.fillRule||"nonzero";n!=="nonzero"&&n!=="evenodd"&&(qt('Fill-rule "'+n+'" is not supported, falling back to "nonzero".'),n="nonzero");let s=n==="nonzero"?(c=>c!==0):(c=>(c&1)!==0),r=[];for(let c of this.subPaths){let l=c.getPoints();if(l.length<3)continue;let u=Un.area(l);if(u===0)continue;let h=new Is;for(let d=0;d<l.length;d++)h.expandByPoint(l[d]);r.push({subPath:c,points:l,boundingBox:h,interiorPoint:e(l,h),absArea:Math.abs(u),winding:u<0?-1:1,container:null,exclude:!1,role:null})}r.sort((c,l)=>l.absArea-c.absArea);for(let c=0;c<r.length;c++){let l=r[c],u=0;for(let h=c-1;h>=0;h--){let d=r[h];if(d.boundingBox.containsBox(l.boundingBox)&&t(l.interiorPoint,d.points)){l.container=d.exclude?d.container:d,u=d.winding,l.winding+=u;break}}s(l.winding)===s(u)&&(l.exclude=!0)}for(let c of r)c.exclude||(c.role=c.container===null||c.container.role==="hole"?"outer":"hole");let a=[],o=new Map;for(let c of r){if(c.exclude||c.role!=="outer")continue;let l=new zi;l.curves=c.subPath.curves,a.push(l),o.set(c,l)}for(let c of r){if(c.exclude||c.role!=="hole")continue;let l=o.get(c.container);if(!l)continue;let u=new On;u.curves=c.subPath.curves,l.holes.push(u)}return a}},Lr=class extends Sn{constructor(t,e=null){super(),this.object=t,this.domElement=e,this.enabled=!0,this.state=-1,this.keys={},this.mouseButtons={LEFT:null,MIDDLE:null,RIGHT:null},this.touches={ONE:null,TWO:null}}connect(t){this.domElement!==null&&this.disconnect(),this.domElement=t}disconnect(){}dispose(){}update(){}};function Ec(i,t,e,n){let s=Ep(n);switch(e){case fc:return i*t;case fo:return i*t/s.components*s.byteLength;case po:return i*t/s.components*s.byteLength;case Ti:return i*t*2/s.components*s.byteLength;case mo:return i*t*2/s.components*s.byteLength;case pc:return i*t*3/s.components*s.byteLength;case Dn:return i*t*4/s.components*s.byteLength;case go:return i*t*4/s.components*s.byteLength;case Ur:case Fr:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*8;case Or:case Br:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*16;case _o:case vo:return Math.max(i,16)*Math.max(t,8)/4;case xo:case yo:return Math.max(i,8)*Math.max(t,8)/2;case bo:case Mo:case wo:case To:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*8;case So:case kr:case Ao:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*16;case Eo:return Math.floor((i+3)/4)*Math.floor((t+3)/4)*16;case Co:return Math.floor((i+4)/5)*Math.floor((t+3)/4)*16;case Ro:return Math.floor((i+4)/5)*Math.floor((t+4)/5)*16;case Po:return Math.floor((i+5)/6)*Math.floor((t+4)/5)*16;case Io:return Math.floor((i+5)/6)*Math.floor((t+5)/6)*16;case Lo:return Math.floor((i+7)/8)*Math.floor((t+4)/5)*16;case Do:return Math.floor((i+7)/8)*Math.floor((t+5)/6)*16;case No:return Math.floor((i+7)/8)*Math.floor((t+7)/8)*16;case Uo:return Math.floor((i+9)/10)*Math.floor((t+4)/5)*16;case Fo:return Math.floor((i+9)/10)*Math.floor((t+5)/6)*16;case Oo:return Math.floor((i+9)/10)*Math.floor((t+7)/8)*16;case Bo:return Math.floor((i+9)/10)*Math.floor((t+9)/10)*16;case ko:return Math.floor((i+11)/12)*Math.floor((t+9)/10)*16;case zo:return Math.floor((i+11)/12)*Math.floor((t+11)/12)*16;case Go:case Vo:case Ho:return Math.ceil(i/4)*Math.ceil(t/4)*16;case Wo:case Xo:return Math.ceil(i/4)*Math.ceil(t/4)*8;case zr:case qo:return Math.ceil(i/4)*Math.ceil(t/4)*16}throw new Error(`Unable to determine texture byte length for ${e} format.`)}function Ep(i){switch(i){case nn:case cc:return{byteLength:1,components:1};case Us:case hc:case Ln:return{byteLength:2,components:1};case ho:case uo:return{byteLength:2,components:4};case In:case co:case xn:return{byteLength:4,components:1};case uc:case dc:return{byteLength:4,components:3}}throw new Error(`THREE.TextureUtils: Unknown texture type ${i}.`)}typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("register",{detail:{revision:"187"}}));typeof window<"u"&&(window.__THREE__?qt("WARNING: Multiple instances of Three.js being imported."):window.__THREE__="187");/**
 * @license
 * Copyright 2010-2026 Three.js Authors
 * SPDX-License-Identifier: MIT
 */function ld(){let i=null,t=!1,e=null,n=null;function s(r,a){n=i.requestAnimationFrame(s),e(r,a)}return{start:function(){t!==!0&&e!==null&&i!==null&&(n=i.requestAnimationFrame(s),t=!0)},stop:function(){i!==null&&i.cancelAnimationFrame(n),t=!1},setAnimationLoop:function(r){e=r},setContext:function(r){i=r}}}function Rp(i){let t=new WeakMap;function e(o,c){let l=o.array,u=o.usage,h=l.byteLength,d=i.createBuffer();i.bindBuffer(c,d),i.bufferData(c,l,u),o.onUploadCallback();let f;if(l instanceof Float32Array)f=i.FLOAT;else if(typeof Float16Array<"u"&&l instanceof Float16Array)f=i.HALF_FLOAT;else if(l instanceof Uint16Array)o.isFloat16BufferAttribute?f=i.HALF_FLOAT:f=i.UNSIGNED_SHORT;else if(l instanceof Int16Array)f=i.SHORT;else if(l instanceof Uint32Array)f=i.UNSIGNED_INT;else if(l instanceof Int32Array)f=i.INT;else if(l instanceof Int8Array)f=i.BYTE;else if(l instanceof Uint8Array)f=i.UNSIGNED_BYTE;else if(l instanceof Uint8ClampedArray)f=i.UNSIGNED_BYTE;else throw new Error("THREE.WebGLAttributes: Unsupported buffer data format: "+l);return{buffer:d,type:f,bytesPerElement:l.BYTES_PER_ELEMENT,version:o.version,size:h}}function n(o,c,l){let u=c.array,h=c.updateRanges;if(i.bindBuffer(l,o),h.length===0)i.bufferSubData(l,0,u);else{h.sort((f,m)=>f.start-m.start);let d=0;for(let f=1;f<h.length;f++){let m=h[d],y=h[f];y.start<=m.start+m.count+1?m.count=Math.max(m.count,y.start+y.count-m.start):(++d,h[d]=y)}h.length=d+1;for(let f=0,m=h.length;f<m;f++){let y=h[f];i.bufferSubData(l,y.start*u.BYTES_PER_ELEMENT,u,y.start,y.count)}c.clearUpdateRanges()}c.onUploadCallback()}function s(o){return o.isInterleavedBufferAttribute&&(o=o.data),t.get(o)}function r(o){o.isInterleavedBufferAttribute&&(o=o.data);let c=t.get(o);c&&(i.deleteBuffer(c.buffer),t.delete(o))}function a(o,c){if(o.isInterleavedBufferAttribute&&(o=o.data),o.isGLBufferAttribute){let u=t.get(o);(!u||u.version<o.version)&&t.set(o,{buffer:o.buffer,type:o.type,bytesPerElement:o.elementSize,version:o.version});return}let l=t.get(o);if(l===void 0)t.set(o,e(o,c));else if(l.version<o.version){if(l.size!==o.array.byteLength)throw new Error("THREE.WebGLAttributes: The size of the buffer attribute's array buffer does not match the original size. Resizing buffer attributes is not supported.");n(l.buffer,o,c),l.version=o.version}}return{get:s,remove:r,update:a}}var Pp=`#ifdef USE_ALPHAHASH
	if ( diffuseColor.a < getAlphaHashThreshold( vPosition ) ) discard;
#endif`,Ip=`#ifdef USE_ALPHAHASH
	const float ALPHA_HASH_SCALE = 0.05;
	float hash2D( vec2 value ) {
		return fract( 1.0e4 * sin( 17.0 * value.x + 0.1 * value.y ) * ( 0.1 + abs( sin( 13.0 * value.y + value.x ) ) ) );
	}
	float hash3D( vec3 value ) {
		return hash2D( vec2( hash2D( value.xy ), value.z ) );
	}
	float getAlphaHashThreshold( vec3 position ) {
		float maxDeriv = max(
			length( dFdx( position.xyz ) ),
			length( dFdy( position.xyz ) )
		);
		float pixScale = 1.0 / ( ALPHA_HASH_SCALE * maxDeriv );
		vec2 pixScales = vec2(
			exp2( floor( log2( pixScale ) ) ),
			exp2( ceil( log2( pixScale ) ) )
		);
		vec2 alpha = vec2(
			hash3D( floor( pixScales.x * position.xyz ) ),
			hash3D( floor( pixScales.y * position.xyz ) )
		);
		float lerpFactor = fract( log2( pixScale ) );
		float x = ( 1.0 - lerpFactor ) * alpha.x + lerpFactor * alpha.y;
		float a = min( lerpFactor, 1.0 - lerpFactor );
		vec3 cases = vec3(
			x * x / ( 2.0 * a * ( 1.0 - a ) ),
			( x - 0.5 * a ) / ( 1.0 - a ),
			1.0 - ( ( 1.0 - x ) * ( 1.0 - x ) / ( 2.0 * a * ( 1.0 - a ) ) )
		);
		float threshold = ( x < ( 1.0 - a ) )
			? ( ( x < a ) ? cases.x : cases.y )
			: cases.z;
		return clamp( threshold , 1.0e-6, 1.0 );
	}
#endif`,Lp=`#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, vAlphaMapUv ).g;
#endif`,Dp=`#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,Np=`#ifdef USE_ALPHATEST
	#ifdef ALPHA_TO_COVERAGE
	diffuseColor.a = smoothstep( alphaTest, alphaTest + fwidth( diffuseColor.a ), diffuseColor.a );
	if ( diffuseColor.a == 0.0 ) discard;
	#else
	if ( diffuseColor.a < alphaTest ) discard;
	#endif
#endif`,Up=`#ifdef USE_ALPHATEST
	uniform float alphaTest;
#endif`,Fp=`#ifdef USE_AOMAP
	float ambientOcclusion = ( texture2D( aoMap, vAoMapUv ).r - 1.0 ) * aoMapIntensity + 1.0;
	reflectedLight.indirectDiffuse *= ambientOcclusion;
	#if defined( USE_CLEARCOAT ) 
		clearcoatSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_SHEEN ) 
		sheenSpecularIndirect *= ambientOcclusion;
	#endif
	#if defined( USE_ENVMAP ) && defined( STANDARD )
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		reflectedLight.indirectSpecular *= computeSpecularOcclusion( dotNV, ambientOcclusion, material.roughness );
	#endif
#endif`,Op=`#ifdef USE_AOMAP
	uniform sampler2D aoMap;
	uniform float aoMapIntensity;
#endif`,Bp=`#ifdef USE_BATCHING
	#if ! defined( GL_ANGLE_multi_draw )
	#define gl_DrawID _gl_DrawID
	uniform int _gl_DrawID;
	#endif
	uniform highp sampler2D batchingTexture;
	uniform highp usampler2D batchingIdTexture;
	mat4 getBatchingMatrix( const in float i ) {
		int size = textureSize( batchingTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( batchingTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( batchingTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( batchingTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( batchingTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
	float getIndirectIndex( const in int i ) {
		int size = textureSize( batchingIdTexture, 0 ).x;
		int x = i % size;
		int y = i / size;
		return float( texelFetch( batchingIdTexture, ivec2( x, y ), 0 ).r );
	}
#endif
#ifdef USE_BATCHING_COLOR
	uniform sampler2D batchingColorTexture;
	vec4 getBatchingColor( const in float i ) {
		int size = textureSize( batchingColorTexture, 0 ).x;
		int j = int( i );
		int x = j % size;
		int y = j / size;
		return texelFetch( batchingColorTexture, ivec2( x, y ), 0 );
	}
#endif`,kp=`#ifdef USE_BATCHING
	mat4 batchingMatrix = getBatchingMatrix( getIndirectIndex( gl_DrawID ) );
#endif`,zp=`vec3 transformed = vec3( position );
#ifdef USE_ALPHAHASH
	vPosition = vec3( position );
#endif`,Gp=`vec3 objectNormal = vec3( normal );
#ifdef USE_TANGENT
	vec3 objectTangent = vec3( tangent.xyz );
#endif`,Vp=`float G_BlinnPhong_Implicit( ) {
	return 0.25;
}
float D_BlinnPhong( const in float shininess, const in float dotNH ) {
	return RECIPROCAL_PI * ( shininess * 0.5 + 1.0 ) * pow( dotNH, shininess );
}
vec3 BRDF_BlinnPhong( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 specularColor, const in float shininess ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( specularColor, 1.0, dotVH );
	float G = G_BlinnPhong_Implicit( );
	float D = D_BlinnPhong( shininess, dotNH );
	return F * ( G * D );
} // validated`,Hp=`#ifdef USE_IRIDESCENCE
	const mat3 XYZ_TO_REC709 = mat3(
		 3.2404542, -0.9692660,  0.0556434,
		-1.5371385,  1.8760108, -0.2040259,
		-0.4985314,  0.0415560,  1.0572252
	);
	vec3 Fresnel0ToIor( vec3 fresnel0 ) {
		vec3 sqrtF0 = sqrt( fresnel0 );
		return ( vec3( 1.0 ) + sqrtF0 ) / ( vec3( 1.0 ) - sqrtF0 );
	}
	vec3 IorToFresnel0( vec3 transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - vec3( incidentIor ) ) / ( transmittedIor + vec3( incidentIor ) ) );
	}
	float IorToFresnel0( float transmittedIor, float incidentIor ) {
		return pow2( ( transmittedIor - incidentIor ) / ( transmittedIor + incidentIor ));
	}
	vec3 evalSensitivity( float OPD, vec3 shift ) {
		float phase = 2.0 * PI * OPD * 1.0e-9;
		vec3 val = vec3( 5.4856e-13, 4.4201e-13, 5.2481e-13 );
		vec3 pos = vec3( 1.6810e+06, 1.7953e+06, 2.2084e+06 );
		vec3 var = vec3( 4.3278e+09, 9.3046e+09, 6.6121e+09 );
		vec3 xyz = val * sqrt( 2.0 * PI * var ) * cos( pos * phase + shift ) * exp( - pow2( phase ) * var );
		xyz.x += 9.7470e-14 * sqrt( 2.0 * PI * 4.5282e+09 ) * cos( 2.2399e+06 * phase + shift[ 0 ] ) * exp( - 4.5282e+09 * pow2( phase ) );
		xyz /= 1.0685e-7;
		vec3 rgb = XYZ_TO_REC709 * xyz;
		return rgb;
	}
	vec3 evalIridescence( float outsideIOR, float eta2, float cosTheta1, float thinFilmThickness, vec3 baseF0 ) {
		vec3 I;
		float iridescenceIOR = mix( outsideIOR, eta2, smoothstep( 0.0, 0.03, thinFilmThickness ) );
		float sinTheta2Sq = pow2( outsideIOR / iridescenceIOR ) * ( 1.0 - pow2( cosTheta1 ) );
		float cosTheta2Sq = 1.0 - sinTheta2Sq;
		if ( cosTheta2Sq < 0.0 ) {
			return vec3( 1.0 );
		}
		float cosTheta2 = sqrt( cosTheta2Sq );
		float R0 = IorToFresnel0( iridescenceIOR, outsideIOR );
		float R12 = F_Schlick( R0, 1.0, cosTheta1 );
		float T121 = 1.0 - R12;
		float phi12 = 0.0;
		if ( iridescenceIOR < outsideIOR ) phi12 = PI;
		float phi21 = PI - phi12;
		vec3 baseIOR = Fresnel0ToIor( clamp( baseF0, 0.0, 0.9999 ) );		vec3 R1 = IorToFresnel0( baseIOR, iridescenceIOR );
		vec3 R23 = F_Schlick( R1, 1.0, cosTheta2 );
		vec3 phi23 = vec3( 0.0 );
		if ( baseIOR[ 0 ] < iridescenceIOR ) phi23[ 0 ] = PI;
		if ( baseIOR[ 1 ] < iridescenceIOR ) phi23[ 1 ] = PI;
		if ( baseIOR[ 2 ] < iridescenceIOR ) phi23[ 2 ] = PI;
		float OPD = 2.0 * iridescenceIOR * thinFilmThickness * cosTheta2;
		vec3 phi = vec3( phi21 ) + phi23;
		vec3 R123 = clamp( R12 * R23, 1e-5, 0.9999 );
		vec3 r123 = sqrt( R123 );
		vec3 Rs = pow2( T121 ) * R23 / ( vec3( 1.0 ) - R123 );
		vec3 C0 = R12 + Rs;
		I = C0;
		vec3 Cm = Rs - T121;
		for ( int m = 1; m <= 2; ++ m ) {
			Cm *= r123;
			vec3 Sm = 2.0 * evalSensitivity( float( m ) * OPD, float( m ) * phi );
			I += Cm * Sm;
		}
		return max( I, vec3( 0.0 ) );
	}
#endif`,Wp=`#ifdef USE_BUMPMAP
	uniform sampler2D bumpMap;
	uniform float bumpScale;
	vec2 dHdxy_fwd() {
		vec2 dSTdx = dFdx( vBumpMapUv );
		vec2 dSTdy = dFdy( vBumpMapUv );
		float Hll = bumpScale * texture2D( bumpMap, vBumpMapUv ).x;
		float dBx = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdx ).x - Hll;
		float dBy = bumpScale * texture2D( bumpMap, vBumpMapUv + dSTdy ).x - Hll;
		return vec2( dBx, dBy );
	}
	vec3 perturbNormalArb( vec3 surf_pos, vec3 surf_norm, vec2 dHdxy, float faceDirection ) {
		vec3 vSigmaX = normalize( dFdx( surf_pos.xyz ) );
		vec3 vSigmaY = normalize( dFdy( surf_pos.xyz ) );
		vec3 vN = surf_norm;
		vec3 R1 = cross( vSigmaY, vN );
		vec3 R2 = cross( vN, vSigmaX );
		float fDet = dot( vSigmaX, R1 ) * faceDirection;
		vec3 vGrad = sign( fDet ) * ( dHdxy.x * R1 + dHdxy.y * R2 );
		return normalize( abs( fDet ) * surf_norm - vGrad );
	}
#endif`,Xp=`#if NUM_CLIPPING_PLANES > 0
	vec4 plane;
	#ifdef ALPHA_TO_COVERAGE
		float distanceToPlane, distanceGradient;
		float clipOpacity = 1.0;
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
			distanceGradient = fwidth( distanceToPlane ) / 2.0;
			clipOpacity *= smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			if ( clipOpacity == 0.0 ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			float unionClipOpacity = 1.0;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				distanceToPlane = - dot( vClipPosition, plane.xyz ) + plane.w;
				distanceGradient = fwidth( distanceToPlane ) / 2.0;
				unionClipOpacity *= 1.0 - smoothstep( - distanceGradient, distanceGradient, distanceToPlane );
			}
			#pragma unroll_loop_end
			clipOpacity *= 1.0 - unionClipOpacity;
		#endif
		diffuseColor.a *= clipOpacity;
		if ( diffuseColor.a == 0.0 ) discard;
	#else
		#pragma unroll_loop_start
		for ( int i = 0; i < UNION_CLIPPING_PLANES; i ++ ) {
			plane = clippingPlanes[ i ];
			if ( dot( vClipPosition, plane.xyz ) > plane.w ) discard;
		}
		#pragma unroll_loop_end
		#if UNION_CLIPPING_PLANES < NUM_CLIPPING_PLANES
			bool clipped = true;
			#pragma unroll_loop_start
			for ( int i = UNION_CLIPPING_PLANES; i < NUM_CLIPPING_PLANES; i ++ ) {
				plane = clippingPlanes[ i ];
				clipped = ( dot( vClipPosition, plane.xyz ) > plane.w ) && clipped;
			}
			#pragma unroll_loop_end
			if ( clipped ) discard;
		#endif
	#endif
#endif`,qp=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
	uniform vec4 clippingPlanes[ NUM_CLIPPING_PLANES ];
#endif`,Yp=`#if NUM_CLIPPING_PLANES > 0
	varying vec3 vClipPosition;
#endif`,$p=`#if NUM_CLIPPING_PLANES > 0
	vClipPosition = - mvPosition.xyz;
#endif`,Zp=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	diffuseColor *= vColor;
#endif`,Jp=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA )
	varying vec4 vColor;
#endif`,Kp=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	varying vec4 vColor;
#endif`,jp=`#if defined( USE_COLOR ) || defined( USE_COLOR_ALPHA ) || defined( USE_INSTANCING_COLOR ) || defined( USE_BATCHING_COLOR )
	vColor = vec4( 1.0 );
#endif
#ifdef USE_COLOR_ALPHA
	vColor *= color;
#elif defined( USE_COLOR )
	vColor.rgb *= color;
#endif
#ifdef USE_INSTANCING_COLOR
	vColor.rgb *= instanceColor.rgb;
#endif
#ifdef USE_BATCHING_COLOR
	vColor *= getBatchingColor( getIndirectIndex( gl_DrawID ) );
#endif`,Qp=`#define PI 3.141592653589793
#define PI2 6.283185307179586
#define PI_HALF 1.5707963267948966
#define RECIPROCAL_PI 0.3183098861837907
#define RECIPROCAL_PI2 0.15915494309189535
#define EPSILON 1e-6
#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
#define whiteComplement( a ) ( 1.0 - saturate( a ) )
float pow2( const in float x ) { return x*x; }
vec3 pow2( const in vec3 x ) { return x*x; }
float pow3( const in float x ) { return x*x*x; }
float pow4( const in float x ) { float x2 = x*x; return x2*x2; }
float max3( const in vec3 v ) { return max( max( v.x, v.y ), v.z ); }
float average( const in vec3 v ) { return dot( v, vec3( 0.3333333 ) ); }
highp float rand( const in vec2 uv ) {
	const highp float a = 12.9898, b = 78.233, c = 43758.5453;
	highp float dt = dot( uv.xy, vec2( a,b ) ), sn = mod( dt, PI );
	return fract( sin( sn ) * c );
}
#ifdef HIGH_PRECISION
	float precisionSafeLength( vec3 v ) { return length( v ); }
#else
	float precisionSafeLength( vec3 v ) {
		float maxComponent = max3( abs( v ) );
		return length( v / maxComponent ) * maxComponent;
	}
#endif
struct IncidentLight {
	vec3 color;
	vec3 direction;
	bool visible;
};
struct ReflectedLight {
	vec3 directDiffuse;
	vec3 directSpecular;
	vec3 indirectDiffuse;
	vec3 indirectSpecular;
};
#ifdef USE_ALPHAHASH
	varying vec3 vPosition;
#endif
vec3 transformDirection( in vec3 dir, in mat4 matrix ) {
	return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );
}
#define inverseTransformDirection transformDirectionByInverseViewMatrix
vec3 transformNormalByInverseViewMatrix( in vec3 normal, in mat4 viewMatrix ) {
	return normalize( ( vec4( normal, 0.0 ) * viewMatrix ).xyz );
}
vec3 transformDirectionByInverseViewMatrix( in vec3 dir, in mat4 viewMatrix ) {
	return normalize( ( vec4( dir, 0.0 ) * viewMatrix ).xyz );
}
bool isPerspectiveMatrix( mat4 m ) {
	return m[ 2 ][ 3 ] == - 1.0;
}
vec2 equirectUv( in vec3 dir ) {
	float u = atan( dir.z, dir.x ) * RECIPROCAL_PI2 + 0.5;
	float v = asin( clamp( dir.y, - 1.0, 1.0 ) ) * RECIPROCAL_PI + 0.5;
	return vec2( u, v );
}
vec3 BRDF_Lambert( const in vec3 diffuseColor ) {
	return RECIPROCAL_PI * diffuseColor;
}
vec3 F_Schlick( const in vec3 f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
}
float F_Schlick( const in float f0, const in float f90, const in float dotVH ) {
	float fresnel = exp2( ( - 5.55473 * dotVH - 6.98316 ) * dotVH );
	return f0 * ( 1.0 - fresnel ) + ( f90 * fresnel );
} // validated`,tm=`vec3 transformedNormal = objectNormal;
#ifdef USE_TANGENT
	vec3 transformedTangent = objectTangent;
#endif
#ifdef USE_BATCHING
	mat3 bm = mat3( batchingMatrix );
	transformedNormal /= vec3( dot( bm[ 0 ], bm[ 0 ] ), dot( bm[ 1 ], bm[ 1 ] ), dot( bm[ 2 ], bm[ 2 ] ) );
	transformedNormal = bm * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = bm * transformedTangent;
	#endif
#endif
#ifdef USE_INSTANCING
	mat3 im = mat3( instanceMatrix );
	transformedNormal /= vec3( dot( im[ 0 ], im[ 0 ] ), dot( im[ 1 ], im[ 1 ] ), dot( im[ 2 ], im[ 2 ] ) );
	transformedNormal = im * transformedNormal;
	#ifdef USE_TANGENT
		transformedTangent = im * transformedTangent;
	#endif
#endif
transformedNormal = normalMatrix * transformedNormal;
#ifdef FLIP_SIDED
	transformedNormal = - transformedNormal;
#endif
#ifdef USE_TANGENT
	transformedTangent = ( modelViewMatrix * vec4( transformedTangent, 0.0 ) ).xyz;
#endif`,em=`#ifdef USE_DISPLACEMENTMAP
	uniform sampler2D displacementMap;
	uniform float displacementScale;
	uniform float displacementBias;
#endif`,nm=`#ifdef USE_DISPLACEMENTMAP
	transformed += normalize( objectNormal ) * ( texture2D( displacementMap, vDisplacementMapUv ).x * displacementScale + displacementBias );
#endif`,im=`#ifdef USE_EMISSIVEMAP
	vec4 emissiveColor = texture2D( emissiveMap, vEmissiveMapUv );
	#ifdef DECODE_VIDEO_TEXTURE_EMISSIVE
		emissiveColor = sRGBTransferEOTF( emissiveColor );
	#endif
	totalEmissiveRadiance *= emissiveColor.rgb;
#endif`,sm=`#ifdef USE_EMISSIVEMAP
	uniform sampler2D emissiveMap;
#endif`,rm="gl_FragColor = linearToOutputTexel( gl_FragColor );",am=`vec4 LinearTransferOETF( in vec4 value ) {
	return value;
}
vec4 sRGBTransferEOTF( in vec4 value ) {
	return vec4( mix( pow( value.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), value.rgb * 0.0773993808, vec3( lessThanEqual( value.rgb, vec3( 0.04045 ) ) ) ), value.a );
}
vec4 sRGBTransferOETF( in vec4 value ) {
	return vec4( mix( pow( value.rgb, vec3( 0.41666 ) ) * 1.055 - vec3( 0.055 ), value.rgb * 12.92, vec3( lessThanEqual( value.rgb, vec3( 0.0031308 ) ) ) ), value.a );
}`,om=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vec3 cameraToFrag;
		if ( isOrthographic ) {
			cameraToFrag = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToFrag = normalize( vWorldPosition - cameraPosition );
		}
		vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vec3 reflectVec = reflect( cameraToFrag, worldNormal );
		#else
			vec3 reflectVec = refract( cameraToFrag, worldNormal, refractionRatio );
		#endif
	#else
		vec3 reflectVec = vReflect;
	#endif
	#ifdef ENVMAP_TYPE_CUBE
		vec4 envColor = textureCube( envMap, envMapRotation * reflectVec );
		#ifdef ENVMAP_BLENDING_MULTIPLY
			outgoingLight = mix( outgoingLight, outgoingLight * envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_MIX )
			outgoingLight = mix( outgoingLight, envColor.xyz, specularStrength * reflectivity );
		#elif defined( ENVMAP_BLENDING_ADD )
			outgoingLight += envColor.xyz * specularStrength * reflectivity;
		#endif
	#endif
#endif`,lm=`#ifdef USE_ENVMAP
	uniform float envMapIntensity;
	uniform mat3 envMapRotation;
	uniform samplerCube envMap;
	#ifdef ENVMAP_TYPE_PMREM
		float roughnessToMip( const in float roughness ) {
			float r = clamp( roughness, 0.0, 1.0 );
			return ENVMAP_MAX_LOD * r * ( 2.0 - r );
		}
	#endif
#endif`,cm=`#ifdef USE_ENVMAP
	uniform float reflectivity;
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		varying vec3 vWorldPosition;
		uniform float refractionRatio;
	#else
		varying vec3 vReflect;
	#endif
#endif`,hm=`#ifdef USE_ENVMAP
	#if defined( USE_BUMPMAP ) || defined( USE_NORMALMAP ) || defined( PHONG ) || defined( LAMBERT )
		#define ENV_WORLDPOS
	#endif
	#ifdef ENV_WORLDPOS
		
		varying vec3 vWorldPosition;
	#else
		varying vec3 vReflect;
		uniform float refractionRatio;
	#endif
#endif`,um=`#ifdef USE_ENVMAP
	#ifdef ENV_WORLDPOS
		vWorldPosition = worldPosition.xyz;
	#else
		vec3 cameraToVertex;
		if ( isOrthographic ) {
			cameraToVertex = normalize( vec3( - viewMatrix[ 0 ][ 2 ], - viewMatrix[ 1 ][ 2 ], - viewMatrix[ 2 ][ 2 ] ) );
		} else {
			cameraToVertex = normalize( worldPosition.xyz - cameraPosition );
		}
		vec3 worldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );
		#ifdef ENVMAP_MODE_REFLECTION
			vReflect = reflect( cameraToVertex, worldNormal );
		#else
			vReflect = refract( cameraToVertex, worldNormal, refractionRatio );
		#endif
	#endif
#endif`,dm=`#ifdef USE_FOG
	vFogDepth = - mvPosition.z;
#endif`,fm=`#ifdef USE_FOG
	varying float vFogDepth;
#endif`,pm=`#ifdef USE_FOG
	#ifdef FOG_EXP2
		float fogFactor = 1.0 - exp( - fogDensity * fogDensity * vFogDepth * vFogDepth );
	#else
		float fogFactor = smoothstep( fogNear, fogFar, vFogDepth );
	#endif
	gl_FragColor.rgb = mix( gl_FragColor.rgb, fogColor, fogFactor );
#endif`,mm=`#ifdef USE_FOG
	uniform vec3 fogColor;
	varying float vFogDepth;
	#ifdef FOG_EXP2
		uniform float fogDensity;
	#else
		uniform float fogNear;
		uniform float fogFar;
	#endif
#endif`,gm=`#ifdef USE_GRADIENTMAP
	uniform sampler2D gradientMap;
#endif
vec3 getGradientIrradiance( vec3 normal, vec3 lightDirection ) {
	float dotNL = dot( normal, lightDirection );
	vec2 coord = vec2( dotNL * 0.5 + 0.5, 0.0 );
	#ifdef USE_GRADIENTMAP
		return vec3( texture2D( gradientMap, coord ).r );
	#else
		vec2 fw = fwidth( coord ) * 0.5;
		return mix( vec3( 0.7 ), vec3( 1.0 ), smoothstep( 0.7 - fw.x, 0.7 + fw.x, coord.x ) );
	#endif
}`,xm=`#ifdef USE_LIGHTMAP
	uniform sampler2D lightMap;
	uniform float lightMapIntensity;
#endif`,_m=`LambertMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularStrength = specularStrength;`,ym=`varying vec3 vViewPosition;
struct LambertMaterial {
	vec3 diffuseColor;
	float specularStrength;
};
void RE_Direct_Lambert( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Lambert( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in LambertMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Lambert
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Lambert`,vm=`uniform bool receiveShadow;
uniform vec3 ambientLightColor;
#if defined( USE_LIGHT_PROBES )
	uniform vec3 lightProbe[ 9 ];
#endif
vec3 shGetIrradianceAt( in vec3 normal, in vec3 shCoefficients[ 9 ] ) {
	float x = normal.x, y = normal.y, z = normal.z;
	vec3 result = shCoefficients[ 0 ] * 0.886227;
	result += shCoefficients[ 1 ] * 2.0 * 0.511664 * y;
	result += shCoefficients[ 2 ] * 2.0 * 0.511664 * z;
	result += shCoefficients[ 3 ] * 2.0 * 0.511664 * x;
	result += shCoefficients[ 4 ] * 2.0 * 0.429043 * x * y;
	result += shCoefficients[ 5 ] * 2.0 * 0.429043 * y * z;
	result += shCoefficients[ 6 ] * ( 0.743125 * z * z - 0.247708 );
	result += shCoefficients[ 7 ] * 2.0 * 0.429043 * x * z;
	result += shCoefficients[ 8 ] * 0.429043 * ( x * x - y * y );
	return result;
}
vec3 getLightProbeIrradiance( const in vec3 lightProbe[ 9 ], const in vec3 normal ) {
	vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
	vec3 irradiance = shGetIrradianceAt( worldNormal, lightProbe );
	return irradiance;
}
vec3 getAmbientLightIrradiance( const in vec3 ambientLightColor ) {
	vec3 irradiance = ambientLightColor;
	return irradiance;
}
float getDistanceAttenuation( const in float lightDistance, const in float cutoffDistance, const in float decayExponent ) {
	float distanceFalloff = 1.0 / max( pow( lightDistance, decayExponent ), 0.01 );
	if ( cutoffDistance > 0.0 ) {
		distanceFalloff *= pow2( saturate( 1.0 - pow4( lightDistance / cutoffDistance ) ) );
	}
	return distanceFalloff;
}
float getSpotAttenuation( const in float coneCosine, const in float penumbraCosine, const in float angleCosine ) {
	return smoothstep( coneCosine, penumbraCosine, angleCosine );
}
#if NUM_SUN_LIGHTS > 0
	struct SunLight {
		vec3 direction;
		vec3 color;
	};
	uniform SunLight sunLights[ NUM_SUN_LIGHTS ];
	void getSunLightInfo( const in SunLight sunLight, out IncidentLight light ) {
		light.color = sunLight.color;
		light.direction = sunLight.direction;
		light.visible = true;
	}
#endif
#if NUM_DIR_LIGHTS > 0
	struct DirectionalLight {
		vec3 direction;
		vec3 color;
	};
	uniform DirectionalLight directionalLights[ NUM_DIR_LIGHTS ];
	void getDirectionalLightInfo( const in DirectionalLight directionalLight, out IncidentLight light ) {
		light.color = directionalLight.color;
		light.direction = directionalLight.direction;
		light.visible = true;
	}
#endif
#if NUM_POINT_LIGHTS > 0
	struct PointLight {
		vec3 position;
		vec3 color;
		float distance;
		float decay;
	};
	uniform PointLight pointLights[ NUM_POINT_LIGHTS ];
	void getPointLightInfo( const in PointLight pointLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = pointLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float lightDistance = length( lVector );
		light.color = pointLight.color;
		light.color *= getDistanceAttenuation( lightDistance, pointLight.distance, pointLight.decay );
		light.visible = ( light.color != vec3( 0.0 ) );
	}
#endif
#if NUM_SPOT_LIGHTS > 0
	struct SpotLight {
		vec3 position;
		vec3 direction;
		vec3 color;
		float distance;
		float decay;
		float coneCos;
		float penumbraCos;
	};
	uniform SpotLight spotLights[ NUM_SPOT_LIGHTS ];
	void getSpotLightInfo( const in SpotLight spotLight, const in vec3 geometryPosition, out IncidentLight light ) {
		vec3 lVector = spotLight.position - geometryPosition;
		light.direction = normalize( lVector );
		float angleCos = dot( light.direction, spotLight.direction );
		float spotAttenuation = getSpotAttenuation( spotLight.coneCos, spotLight.penumbraCos, angleCos );
		if ( spotAttenuation > 0.0 ) {
			float lightDistance = length( lVector );
			light.color = spotLight.color * spotAttenuation;
			light.color *= getDistanceAttenuation( lightDistance, spotLight.distance, spotLight.decay );
			light.visible = ( light.color != vec3( 0.0 ) );
		} else {
			light.color = vec3( 0.0 );
			light.visible = false;
		}
	}
#endif
#if NUM_RECT_AREA_LIGHTS > 0
	struct RectAreaLight {
		vec3 color;
		vec3 position;
		vec3 halfWidth;
		vec3 halfHeight;
	};
	uniform sampler2D ltc_1;	uniform sampler2D ltc_2;
	uniform RectAreaLight rectAreaLights[ NUM_RECT_AREA_LIGHTS ];
#endif
#if NUM_HEMI_LIGHTS > 0
	struct HemisphereLight {
		vec3 direction;
		vec3 skyColor;
		vec3 groundColor;
	};
	uniform HemisphereLight hemisphereLights[ NUM_HEMI_LIGHTS ];
	vec3 getHemisphereLightIrradiance( const in HemisphereLight hemiLight, const in vec3 normal ) {
		float dotNL = dot( normal, hemiLight.direction );
		float hemiDiffuseWeight = 0.5 * dotNL + 0.5;
		vec3 irradiance = mix( hemiLight.groundColor, hemiLight.skyColor, hemiDiffuseWeight );
		return irradiance;
	}
#endif
#include <lightprobes_pars_fragment>`,bm=`#ifdef USE_ENVMAP
	vec3 getIBLIrradiance( const in vec3 normal ) {
		#ifdef ENVMAP_TYPE_PMREM
			vec3 worldNormal = transformNormalByInverseViewMatrix( normal, viewMatrix );
			vec4 envMapColor = textureLod( envMap, envMapRotation * worldNormal, ENVMAP_MAX_LOD );
			return PI * envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	vec3 getIBLRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
		#ifdef ENVMAP_TYPE_PMREM
			vec3 reflectVec = reflect( - viewDir, normal );
			reflectVec = normalize( mix( reflectVec, normal, pow4( roughness ) ) );
			reflectVec = transformDirectionByInverseViewMatrix( reflectVec, viewMatrix );
			vec4 envMapColor = textureLod( envMap, envMapRotation * reflectVec, roughnessToMip( roughness ) );
			return envMapColor.rgb * envMapIntensity;
		#else
			return vec3( 0.0 );
		#endif
	}
	#ifdef USE_RETROREFLECTION
		vec3 getIBLRetroRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness ) {
			#ifdef ENVMAP_TYPE_PMREM
				vec3 retroVec = normalize( mix( viewDir, normal, pow4( roughness ) ) );
				retroVec = transformDirectionByInverseViewMatrix( retroVec, viewMatrix );
				vec4 envMapColor = textureLod( envMap, envMapRotation * retroVec, roughnessToMip( roughness ) );
				return envMapColor.rgb * envMapIntensity;
			#else
				return vec3( 0.0 );
			#endif
		}
	#endif
	#ifdef USE_ANISOTROPY
		vec3 getIBLAnisotropyRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
			#ifdef ENVMAP_TYPE_PMREM
				vec3 bentNormal = cross( bitangent, viewDir );
				bentNormal = normalize( cross( bentNormal, bitangent ) );
				bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
				return getIBLRadiance( viewDir, bentNormal, roughness );
			#else
				return vec3( 0.0 );
			#endif
		}
		#ifdef USE_RETROREFLECTION
			vec3 getIBLAnisotropyRetroRadiance( const in vec3 viewDir, const in vec3 normal, const in float roughness, const in vec3 bitangent, const in float anisotropy ) {
				#ifdef ENVMAP_TYPE_PMREM
					vec3 bentNormal = cross( bitangent, viewDir );
					bentNormal = normalize( cross( bentNormal, bitangent ) );
					bentNormal = normalize( mix( bentNormal, normal, pow2( pow2( 1.0 - anisotropy * ( 1.0 - roughness ) ) ) ) );
					return getIBLRetroRadiance( viewDir, bentNormal, roughness );
				#else
					return vec3( 0.0 );
				#endif
			}
		#endif
	#endif
#endif`,Mm=`ToonMaterial material;
material.diffuseColor = diffuseColor.rgb;`,Sm=`varying vec3 vViewPosition;
struct ToonMaterial {
	vec3 diffuseColor;
};
void RE_Direct_Toon( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 irradiance = getGradientIrradiance( geometryNormal, directLight.direction ) * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
void RE_IndirectDiffuse_Toon( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in ToonMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_Toon
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Toon`,wm=`BlinnPhongMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.specularColor = specular;
material.specularShininess = shininess;
material.specularStrength = specularStrength;`,Tm=`varying vec3 vViewPosition;
struct BlinnPhongMaterial {
	vec3 diffuseColor;
	vec3 specularColor;
	float specularShininess;
	float specularStrength;
};
void RE_Direct_BlinnPhong( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	reflectedLight.directDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
	reflectedLight.directSpecular += irradiance * BRDF_BlinnPhong( directLight.direction, geometryViewDir, geometryNormal, material.specularColor, material.specularShininess ) * material.specularStrength;
}
void RE_IndirectDiffuse_BlinnPhong( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in BlinnPhongMaterial material, inout ReflectedLight reflectedLight ) {
	reflectedLight.indirectDiffuse += irradiance * BRDF_Lambert( material.diffuseColor );
}
#define RE_Direct				RE_Direct_BlinnPhong
#define RE_IndirectDiffuse		RE_IndirectDiffuse_BlinnPhong`,Am=`PhysicalMaterial material;
material.diffuseColor = diffuseColor.rgb;
material.diffuseContribution = diffuseColor.rgb * ( 1.0 - metalnessFactor );
material.metalness = metalnessFactor;
vec3 dxy = max( abs( dFdx( nonPerturbedNormal ) ), abs( dFdy( nonPerturbedNormal ) ) );
float geometryRoughness = max( max( dxy.x, dxy.y ), dxy.z );
material.roughness = max( roughnessFactor, 0.045 );
material.roughness += geometryRoughness;
material.roughness = min( material.roughness, 1.0 );
#ifdef USE_DIFFUSE_ROUGHNESS
	float diffuseRoughnessFactor = diffuseRoughness;
	#ifdef USE_DIFFUSE_ROUGHNESSMAP
		diffuseRoughnessFactor *= texture2D( diffuseRoughnessMap, vDiffuseRoughnessMapUv ).r;
	#endif
	material.diffuseRoughness = saturate( diffuseRoughnessFactor );
#endif
#ifdef IOR
	material.ior = ior;
	#ifdef USE_SPECULAR
		float specularIntensityFactor = specularIntensity;
		vec3 specularColorFactor = specularColor;
		#ifdef USE_SPECULAR_COLORMAP
			specularColorFactor *= texture2D( specularColorMap, vSpecularColorMapUv ).rgb;
		#endif
		#ifdef USE_SPECULAR_INTENSITYMAP
			specularIntensityFactor *= texture2D( specularIntensityMap, vSpecularIntensityMapUv ).a;
		#endif
		material.specularF90 = mix( specularIntensityFactor, 1.0, metalnessFactor );
	#else
		float specularIntensityFactor = 1.0;
		vec3 specularColorFactor = vec3( 1.0 );
		material.specularF90 = 1.0;
	#endif
	material.specularColor = min( pow2( ( material.ior - 1.0 ) / ( material.ior + 1.0 ) ) * specularColorFactor, vec3( 1.0 ) ) * specularIntensityFactor;
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
#else
	material.specularColor = vec3( 0.04 );
	material.specularColorBlended = mix( material.specularColor, diffuseColor.rgb, metalnessFactor );
	material.specularF90 = 1.0;
#endif
#ifdef USE_CLEARCOAT
	material.clearcoat = clearcoat;
	material.clearcoatRoughness = clearcoatRoughness;
	material.clearcoatF0 = vec3( 0.04 );
	material.clearcoatF90 = 1.0;
	#ifdef USE_CLEARCOATMAP
		material.clearcoat *= texture2D( clearcoatMap, vClearcoatMapUv ).x;
	#endif
	#ifdef USE_CLEARCOAT_ROUGHNESSMAP
		material.clearcoatRoughness *= texture2D( clearcoatRoughnessMap, vClearcoatRoughnessMapUv ).y;
	#endif
	material.clearcoat = saturate( material.clearcoat );	material.clearcoatRoughness = max( material.clearcoatRoughness, 0.045 );
	material.clearcoatRoughness += geometryRoughness;
	material.clearcoatRoughness = min( material.clearcoatRoughness, 1.0 );
#endif
#ifdef USE_DISPERSION
	material.dispersion = dispersion;
#endif
#ifdef USE_RETROREFLECTION
	material.retroreflectivity = retroreflectivity;
#endif
#ifdef USE_IRIDESCENCE
	material.iridescence = iridescence;
	material.iridescenceIOR = iridescenceIOR;
	#ifdef USE_IRIDESCENCEMAP
		material.iridescence *= texture2D( iridescenceMap, vIridescenceMapUv ).r;
	#endif
	#ifdef USE_IRIDESCENCE_THICKNESSMAP
		material.iridescenceThickness = (iridescenceThicknessMaximum - iridescenceThicknessMinimum) * texture2D( iridescenceThicknessMap, vIridescenceThicknessMapUv ).g + iridescenceThicknessMinimum;
	#else
		material.iridescenceThickness = iridescenceThicknessMaximum;
	#endif
#endif
#ifdef USE_SHEEN
	material.sheenColor = sheenColor;
	#ifdef USE_SHEEN_COLORMAP
		material.sheenColor *= texture2D( sheenColorMap, vSheenColorMapUv ).rgb;
	#endif
	material.sheenRoughness = clamp( sheenRoughness, 0.0001, 1.0 );
	#ifdef USE_SHEEN_ROUGHNESSMAP
		material.sheenRoughness *= texture2D( sheenRoughnessMap, vSheenRoughnessMapUv ).a;
	#endif
#endif
#ifdef USE_ANISOTROPY
	#ifdef USE_ANISOTROPYMAP
		mat2 anisotropyMat = mat2( anisotropyVector.x, anisotropyVector.y, - anisotropyVector.y, anisotropyVector.x );
		vec3 anisotropyPolar = texture2D( anisotropyMap, vAnisotropyMapUv ).rgb;
		vec2 anisotropyV = anisotropyMat * normalize( 2.0 * anisotropyPolar.rg - vec2( 1.0 ) ) * anisotropyPolar.b;
	#else
		vec2 anisotropyV = anisotropyVector;
	#endif
	material.anisotropy = length( anisotropyV );
	if( material.anisotropy == 0.0 ) {
		anisotropyV = vec2( 1.0, 0.0 );
	} else {
		anisotropyV /= material.anisotropy;
		material.anisotropy = saturate( material.anisotropy );
	}
	material.alphaT = mix( pow2( material.roughness ), 1.0, pow2( material.anisotropy ) );
	material.anisotropyT = tbn[ 0 ] * anisotropyV.x + tbn[ 1 ] * anisotropyV.y;
	material.anisotropyB = tbn[ 1 ] * anisotropyV.x - tbn[ 0 ] * anisotropyV.y;
#endif`,Em=`uniform sampler2D dfgLUT;
struct PhysicalMaterial {
	vec3 diffuseColor;
	vec3 diffuseContribution;
	vec3 specularColor;
	vec3 specularColorBlended;
	float roughness;
	float metalness;
	float specularF90;
	float dispersion;
	vec2 dfg;
	vec3 multiScatteringCompensation;
	#ifdef USE_DIFFUSE_ROUGHNESS
		float diffuseRoughness;
	#endif
	#ifdef USE_RETROREFLECTION
		float retroreflectivity;
	#endif
	#ifdef USE_CLEARCOAT
		float clearcoat;
		float clearcoatRoughness;
		vec3 clearcoatF0;
		float clearcoatF90;
	#endif
	#ifdef USE_IRIDESCENCE
		float iridescence;
		float iridescenceIOR;
		float iridescenceThickness;
		vec3 iridescenceFresnel;
		vec3 iridescenceF0Dielectric;
		vec3 iridescenceF0Metallic;
	#endif
	#ifdef USE_SHEEN
		vec3 sheenColor;
		float sheenRoughness;
	#endif
	#ifdef IOR
		float ior;
	#endif
	#ifdef USE_TRANSMISSION
		float transmission;
		float transmissionAlpha;
		float thickness;
		float attenuationDistance;
		vec3 attenuationColor;
	#endif
	#ifdef USE_ANISOTROPY
		float anisotropy;
		float alphaT;
		vec3 anisotropyT;
		vec3 anisotropyB;
	#endif
};
vec3 clearcoatSpecularDirect = vec3( 0.0 );
vec3 clearcoatSpecularIndirect = vec3( 0.0 );
vec3 sheenSpecularDirect = vec3( 0.0 );
vec3 sheenSpecularIndirect = vec3(0.0 );
vec3 Schlick_to_F0( const in vec3 f, const in float f90, const in float dotVH ) {
    float x = clamp( 1.0 - dotVH, 0.0, 1.0 );
    float x2 = x * x;
    float x5 = clamp( x * x2 * x2, 0.0, 0.9999 );
    return ( f - vec3( f90 ) * x5 ) / ( 1.0 - x5 );
}
#ifdef USE_DIFFUSE_ROUGHNESS
	const float EON_EPSILON = 1e-7;
	float FON_DirectionalAlbedo( const in float mu, const in float roughness, const in float A ) {
		float muComp = 1.0 - mu;
		float gOverPi = muComp * ( 0.0571085289 + muComp * ( 0.491881867 + muComp * ( - 0.332181442 + muComp * 0.0714429953 ) ) );
		return A * ( 1.0 + roughness * gOverPi );
	}
	vec3 BRDF_EON( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in vec3 diffuseColor, const in float roughness ) {
		vec3 rho = saturate( diffuseColor );
		if ( roughness <= EON_EPSILON ) return BRDF_Lambert( rho );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float s = dot( lightDir, viewDir ) - dotNL * dotNV;
		float sOverT = ( s > 0.0 ) ? s / max( max( dotNL, dotNV ), EON_EPSILON ) : s;
		float A = 1.0 / ( 1.0 + ( 0.5 - 2.0 / ( 3.0 * PI ) ) * roughness );
		vec3 singleScatter = rho * RECIPROCAL_PI * A * ( 1.0 + roughness * sOverT );
		float averageAlbedo = A * ( 1.0 + ( 2.0 / 3.0 - 28.0 / ( 15.0 * PI ) ) * roughness );
		float albedoV = FON_DirectionalAlbedo( dotNV, roughness, A );
		float albedoL = FON_DirectionalAlbedo( dotNL, roughness, A );
		vec3 rhoMultiScatter = rho * rho * averageAlbedo / max( vec3( EON_EPSILON ), vec3( 1.0 ) - rho * ( 1.0 - averageAlbedo ) );
		vec3 multiScatter = rhoMultiScatter * RECIPROCAL_PI
			* max( EON_EPSILON, 1.0 - albedoV )
			* max( EON_EPSILON, 1.0 - albedoL )
			/ max( EON_EPSILON, 1.0 - averageAlbedo );
		return singleScatter + multiScatter;
	}
	vec3 EON_DirectionalAlbedo( const in vec3 diffuseColor, const in float roughness, const in float dotNV ) {
		vec3 rho = saturate( diffuseColor );
		if ( roughness <= EON_EPSILON ) return rho;
		float A = 1.0 / ( 1.0 + ( 0.5 - 2.0 / ( 3.0 * PI ) ) * roughness );
		float directionalAlbedo = FON_DirectionalAlbedo( dotNV, roughness, A );
		float averageAlbedo = A * ( 1.0 + ( 2.0 / 3.0 - 28.0 / ( 15.0 * PI ) ) * roughness );
		vec3 rhoMultiScatter = rho * rho * averageAlbedo / max( vec3( EON_EPSILON ), vec3( 1.0 ) - rho * ( 1.0 - averageAlbedo ) );
		return rho * directionalAlbedo + rhoMultiScatter * ( 1.0 - directionalAlbedo );
	}
#endif
float V_GGX_SmithCorrelated( const in float alpha, const in float dotNL, const in float dotNV ) {
	float a2 = pow2( alpha );
	float gv = dotNL * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNV ) );
	float gl = dotNV * sqrt( a2 + ( 1.0 - a2 ) * pow2( dotNL ) );
	return 0.5 / max( gv + gl, EPSILON );
}
float D_GGX( const in float alpha, const in float dotNH ) {
	float a2 = pow2( alpha );
	float denom = pow2( dotNH ) * ( a2 - 1.0 ) + 1.0;
	return RECIPROCAL_PI * a2 / pow2( denom );
}
#ifdef USE_ANISOTROPY
	float V_GGX_SmithCorrelated_Anisotropic( const in float alphaT, const in float alphaB, const in float dotTV, const in float dotBV, const in float dotTL, const in float dotBL, const in float dotNV, const in float dotNL ) {
		float gv = dotNL * length( vec3( alphaT * dotTV, alphaB * dotBV, dotNV ) );
		float gl = dotNV * length( vec3( alphaT * dotTL, alphaB * dotBL, dotNL ) );
		return 0.5 / max( gv + gl, EPSILON );
	}
	float D_GGX_Anisotropic( const in float alphaT, const in float alphaB, const in float dotNH, const in float dotTH, const in float dotBH ) {
		float a2 = alphaT * alphaB;
		highp vec3 v = vec3( alphaB * dotTH, alphaT * dotBH, a2 * dotNH );
		highp float v2 = dot( v, v );
		float w2 = a2 / v2;
		return RECIPROCAL_PI * a2 * pow2 ( w2 );
	}
#endif
#ifdef USE_CLEARCOAT
	vec3 BRDF_GGX_Clearcoat( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material) {
		vec3 f0 = material.clearcoatF0;
		float f90 = material.clearcoatF90;
		float roughness = max( material.clearcoatRoughness, 0.045 );
		float alpha = pow2( roughness );
		vec3 halfDir = normalize( lightDir + viewDir );
		float dotNL = saturate( dot( normal, lightDir ) );
		float dotNV = saturate( dot( normal, viewDir ) );
		float dotNH = saturate( dot( normal, halfDir ) );
		float dotVH = saturate( dot( viewDir, halfDir ) );
		vec3 F = F_Schlick( f0, f90, dotVH );
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
		return F * ( V * D );
	}
#endif
vec3 BRDF_GGX( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, const in PhysicalMaterial material ) {
	vec3 f0 = material.specularColorBlended;
	float f90 = material.specularF90;
	float roughness = max( material.roughness, 0.045 );
	float alpha = pow2( roughness );
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float dotVH = saturate( dot( viewDir, halfDir ) );
	vec3 F = F_Schlick( f0, f90, dotVH );
	#ifdef USE_IRIDESCENCE
		F = mix( F, material.iridescenceFresnel, material.iridescence );
	#endif
	#ifdef USE_ANISOTROPY
		float dotTL = dot( material.anisotropyT, lightDir );
		float dotTV = dot( material.anisotropyT, viewDir );
		float dotTH = dot( material.anisotropyT, halfDir );
		float dotBL = dot( material.anisotropyB, lightDir );
		float dotBV = dot( material.anisotropyB, viewDir );
		float dotBH = dot( material.anisotropyB, halfDir );
		float alphaT = max( material.alphaT, alpha );
		float V = V_GGX_SmithCorrelated_Anisotropic( alphaT, alpha, dotTV, dotBV, dotTL, dotBL, dotNV, dotNL );
		float D = D_GGX_Anisotropic( alphaT, alpha, dotNH, dotTH, dotBH );
	#else
		float V = V_GGX_SmithCorrelated( alpha, dotNL, dotNV );
		float D = D_GGX( alpha, dotNH );
	#endif
	return F * ( V * D );
}
vec2 LTC_Uv( const in vec3 N, const in vec3 V, const in float roughness ) {
	const float LUT_SIZE = 64.0;
	const float LUT_SCALE = ( LUT_SIZE - 1.0 ) / LUT_SIZE;
	const float LUT_BIAS = 0.5 / LUT_SIZE;
	float dotNV = saturate( dot( N, V ) );
	vec2 uv = vec2( roughness, sqrt( 1.0 - dotNV ) );
	uv = uv * LUT_SCALE + LUT_BIAS;
	return uv;
}
float LTC_ClippedSphereFormFactor( const in vec3 f ) {
	float l = length( f );
	return max( ( l * l + f.z ) / ( l + 1.0 ), 0.0 );
}
vec3 LTC_EdgeVectorFormFactor( const in vec3 v1, const in vec3 v2 ) {
	float x = dot( v1, v2 );
	float y = abs( x );
	float a = 0.8543985 + ( 0.4965155 + 0.0145206 * y ) * y;
	float b = 3.4175940 + ( 4.1616724 + y ) * y;
	float v = a / b;
	float theta_sintheta = ( x > 0.0 ) ? v : 0.5 * inversesqrt( max( 1.0 - x * x, 1e-7 ) ) - v;
	return cross( v1, v2 ) * theta_sintheta;
}
vec3 LTC_Evaluate( const in vec3 N, const in vec3 V, const in vec3 P, const in mat3 mInv, const in vec3 rectCoords[ 4 ] ) {
	vec3 v1 = rectCoords[ 1 ] - rectCoords[ 0 ];
	vec3 v2 = rectCoords[ 3 ] - rectCoords[ 0 ];
	vec3 lightNormal = cross( v1, v2 );
	if( dot( lightNormal, P - rectCoords[ 0 ] ) < 0.0 ) return vec3( 0.0 );
	vec3 T1, T2;
	T1 = normalize( V - N * dot( V, N ) );
	T2 = - cross( N, T1 );
	mat3 mat = mInv * transpose( mat3( T1, T2, N ) );
	vec3 coords[ 4 ];
	coords[ 0 ] = mat * ( rectCoords[ 0 ] - P );
	coords[ 1 ] = mat * ( rectCoords[ 1 ] - P );
	coords[ 2 ] = mat * ( rectCoords[ 2 ] - P );
	coords[ 3 ] = mat * ( rectCoords[ 3 ] - P );
	coords[ 0 ] = normalize( coords[ 0 ] );
	coords[ 1 ] = normalize( coords[ 1 ] );
	coords[ 2 ] = normalize( coords[ 2 ] );
	coords[ 3 ] = normalize( coords[ 3 ] );
	vec3 vectorFormFactor = vec3( 0.0 );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 0 ], coords[ 1 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 1 ], coords[ 2 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 2 ], coords[ 3 ] );
	vectorFormFactor += LTC_EdgeVectorFormFactor( coords[ 3 ], coords[ 0 ] );
	float result = LTC_ClippedSphereFormFactor( vectorFormFactor );
	return vec3( result );
}
#if defined( USE_SHEEN )
float D_Charlie( float roughness, float dotNH ) {
	float alpha = pow2( roughness );
	float invAlpha = 1.0 / alpha;
	float cos2h = dotNH * dotNH;
	float sin2h = max( 1.0 - cos2h, 0.0078125 );
	return ( 2.0 + invAlpha ) * pow( sin2h, invAlpha * 0.5 ) / ( 2.0 * PI );
}
float V_Neubelt( float dotNV, float dotNL ) {
	return saturate( 1.0 / ( 4.0 * ( dotNL + dotNV - dotNL * dotNV ) ) );
}
vec3 BRDF_Sheen( const in vec3 lightDir, const in vec3 viewDir, const in vec3 normal, vec3 sheenColor, const in float sheenRoughness ) {
	vec3 halfDir = normalize( lightDir + viewDir );
	float dotNL = saturate( dot( normal, lightDir ) );
	float dotNV = saturate( dot( normal, viewDir ) );
	float dotNH = saturate( dot( normal, halfDir ) );
	float D = D_Charlie( sheenRoughness, dotNH );
	float V = V_Neubelt( dotNV, dotNL );
	return sheenColor * ( D * V );
}
#endif
float IBLSheenBRDF( const in vec3 normal, const in vec3 viewDir, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	float r2 = roughness * roughness;
	float rInv = 1.0 / ( roughness + 0.1 );
	float a = -1.9362 + 1.0678 * roughness + 0.4573 * r2 - 0.8469 * rInv;
	float b = -0.6014 + 0.5538 * roughness - 0.4670 * r2 - 0.1255 * rInv;
	float DG = exp( a * dotNV + b );
	return saturate( DG );
}
vec3 EnvironmentBRDF( const in vec3 normal, const in vec3 viewDir, const in vec3 specularColor, const in float specularF90, const in float roughness ) {
	float dotNV = saturate( dot( normal, viewDir ) );
	vec2 fab = texture2D( dfgLUT, vec2( roughness, dotNV ) ).rg;
	return specularColor * fab.x + specularF90 * fab.y;
}
#ifdef USE_IRIDESCENCE
void computeMultiscatteringIridescence( const in vec2 fab, const in vec3 specularColor, const in float specularF90, const in float iridescence, const in vec3 iridescenceF0, inout vec3 singleScatter, inout vec3 multiScatter ) {
#else
void computeMultiscattering( const in vec2 fab, const in vec3 specularColor, const in float specularF90, inout vec3 singleScatter, inout vec3 multiScatter ) {
#endif
	#ifdef USE_IRIDESCENCE
		vec3 Fr = mix( specularColor, iridescenceF0, iridescence );
	#else
		vec3 Fr = specularColor;
	#endif
	vec3 FssEss = Fr * fab.x + specularF90 * fab.y;
	float Ess = fab.x + fab.y;
	float Ems = 1.0 - Ess;
	vec3 Favg = Fr + ( 1.0 - Fr ) * 0.047619;	vec3 Fms = FssEss * Favg / ( 1.0 - Ems * Favg );
	singleScatter += FssEss;
	multiScatter += Fms * Ems;
}
#if NUM_RECT_AREA_LIGHTS > 0
	void RE_Direct_RectArea_Physical( const in RectAreaLight rectAreaLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
		vec3 normal = geometryNormal;
		vec3 viewDir = geometryViewDir;
		vec3 position = geometryPosition;
		vec3 lightPos = rectAreaLight.position;
		vec3 halfWidth = rectAreaLight.halfWidth;
		vec3 halfHeight = rectAreaLight.halfHeight;
		vec3 lightColor = rectAreaLight.color;
		float roughness = material.roughness;
		vec3 rectCoords[ 4 ];
		rectCoords[ 0 ] = lightPos + halfWidth - halfHeight;		rectCoords[ 1 ] = lightPos - halfWidth - halfHeight;
		rectCoords[ 2 ] = lightPos - halfWidth + halfHeight;
		rectCoords[ 3 ] = lightPos + halfWidth + halfHeight;
		vec2 uv = LTC_Uv( normal, viewDir, roughness );
		vec4 t1 = texture2D( ltc_1, uv );
		vec4 t2 = texture2D( ltc_2, uv );
		mat3 mInv = mat3(
			vec3( t1.x, 0, t1.y ),
			vec3(    0, 1,    0 ),
			vec3( t1.z, 0, t1.w )
		);
		vec3 fresnel = ( material.specularColorBlended * t2.x + ( material.specularF90 - material.specularColorBlended ) * t2.y );
		reflectedLight.directSpecular += lightColor * fresnel * LTC_Evaluate( normal, viewDir, position, mInv, rectCoords );
		reflectedLight.directDiffuse += lightColor * material.diffuseContribution * LTC_Evaluate( normal, viewDir, position, mat3( 1.0 ), rectCoords );
		#ifdef USE_CLEARCOAT
			vec3 Ncc = geometryClearcoatNormal;
			vec2 uvClearcoat = LTC_Uv( Ncc, viewDir, material.clearcoatRoughness );
			vec4 t1Clearcoat = texture2D( ltc_1, uvClearcoat );
			vec4 t2Clearcoat = texture2D( ltc_2, uvClearcoat );
			mat3 mInvClearcoat = mat3(
				vec3( t1Clearcoat.x, 0, t1Clearcoat.y ),
				vec3(             0, 1,             0 ),
				vec3( t1Clearcoat.z, 0, t1Clearcoat.w )
			);
			vec3 fresnelClearcoat = material.clearcoatF0 * t2Clearcoat.x + ( material.clearcoatF90 - material.clearcoatF0 ) * t2Clearcoat.y;
			clearcoatSpecularDirect += lightColor * fresnelClearcoat * LTC_Evaluate( Ncc, viewDir, position, mInvClearcoat, rectCoords );
		#endif
	}
#endif
void RE_Direct_Physical( const in IncidentLight directLight, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	float dotNL = saturate( dot( geometryNormal, directLight.direction ) );
	vec3 irradiance = dotNL * directLight.color;
	#ifdef USE_CLEARCOAT
		float dotNLcc = saturate( dot( geometryClearcoatNormal, directLight.direction ) );
		vec3 ccIrradiance = dotNLcc * directLight.color;
		clearcoatSpecularDirect += ccIrradiance * BRDF_GGX_Clearcoat( directLight.direction, geometryViewDir, geometryClearcoatNormal, material );
	#endif
	#ifdef USE_SHEEN
 
 		sheenSpecularDirect += irradiance * BRDF_Sheen( directLight.direction, geometryViewDir, geometryNormal, material.sheenColor, material.sheenRoughness );
 
 		float sheenAlbedoV = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
 		float sheenAlbedoL = IBLSheenBRDF( geometryNormal, directLight.direction, material.sheenRoughness );
 
 		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * max( sheenAlbedoV, sheenAlbedoL );
 
 		irradiance *= sheenEnergyComp;
 
 	#endif
	vec3 specularBRDF = BRDF_GGX( directLight.direction, geometryViewDir, geometryNormal, material );
	#ifdef USE_RETROREFLECTION
		vec3 retroViewDir = reflect( - geometryViewDir, geometryNormal );
		vec3 retroSpecularBRDF = BRDF_GGX( directLight.direction, retroViewDir, geometryNormal, material );
		specularBRDF = mix( specularBRDF, retroSpecularBRDF, saturate( material.retroreflectivity ) );
	#endif
	reflectedLight.directSpecular += irradiance * specularBRDF * material.multiScatteringCompensation;
	vec3 halfDir = normalize( directLight.direction + geometryViewDir );
	float dotVH = saturate( dot( geometryViewDir, halfDir ) );
	vec3 F = F_Schlick( material.specularColor, material.specularF90, dotVH );
	#ifdef USE_RETROREFLECTION
		vec3 retroHalfDir = normalize( directLight.direction + retroViewDir );
		float dotRetroVH = saturate( dot( retroViewDir, retroHalfDir ) );
		vec3 retroF = F_Schlick( material.specularColor, material.specularF90, dotRetroVH );
		F = mix( F, retroF, saturate( material.retroreflectivity ) );
	#endif
	#ifdef USE_DIFFUSE_ROUGHNESS
		vec3 diffuseBRDF = BRDF_EON( directLight.direction, geometryViewDir, geometryNormal, material.diffuseColor, material.diffuseRoughness ) * ( 1.0 - material.metalness );
	#else
		vec3 diffuseBRDF = BRDF_Lambert( material.diffuseContribution );
	#endif
	reflectedLight.directDiffuse += irradiance * diffuseBRDF * ( 1.0 - F );
}
void RE_IndirectDiffuse_Physical( const in vec3 irradiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight ) {
	vec3 singleScattering = vec3( 0.0 );
	vec3 multiScattering = vec3( 0.0 );
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( material.dfg, material.specularColor, material.specularF90, material.iridescence, material.iridescenceF0Dielectric, singleScattering, multiScattering );
	#else
		computeMultiscattering( material.dfg, material.specularColor, material.specularF90, singleScattering, multiScattering );
	#endif
	#ifdef USE_DIFFUSE_ROUGHNESS
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		vec3 diffuseAlbedo = EON_DirectionalAlbedo( material.diffuseColor, material.diffuseRoughness, dotNV ) * ( 1.0 - material.metalness );
		vec3 diffuse = irradiance * RECIPROCAL_PI * diffuseAlbedo * ( 1.0 - singleScattering - multiScattering );
	#else
		vec3 diffuse = irradiance * BRDF_Lambert( material.diffuseContribution ) * ( 1.0 - singleScattering - multiScattering );
	#endif
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		sheenSpecularIndirect += irradiance * material.sheenColor * sheenAlbedo * RECIPROCAL_PI;
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		diffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectDiffuse += diffuse;
}
void RE_IndirectSpecular_Physical( const in vec3 radiance, const in vec3 irradiance, const in vec3 clearcoatRadiance, const in vec3 geometryPosition, const in vec3 geometryNormal, const in vec3 geometryViewDir, const in vec3 geometryClearcoatNormal, const in PhysicalMaterial material, inout ReflectedLight reflectedLight) {
	#ifdef USE_CLEARCOAT
		clearcoatSpecularIndirect += clearcoatRadiance * EnvironmentBRDF( geometryClearcoatNormal, geometryViewDir, material.clearcoatF0, material.clearcoatF90, material.clearcoatRoughness );
	#endif
	#ifdef USE_SHEEN
		sheenSpecularIndirect += irradiance * material.sheenColor * IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness ) * RECIPROCAL_PI;
 	#endif
	vec3 singleScatteringDielectric = vec3( 0.0 );
	vec3 multiScatteringDielectric = vec3( 0.0 );
	vec3 singleScatteringMetallic = vec3( 0.0 );
	vec3 multiScatteringMetallic = vec3( 0.0 );
	#ifdef USE_IRIDESCENCE
		computeMultiscatteringIridescence( material.dfg, material.specularColor, material.specularF90, material.iridescence, material.iridescenceF0Dielectric, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscatteringIridescence( material.dfg, material.diffuseColor, material.specularF90, material.iridescence, material.iridescenceF0Metallic, singleScatteringMetallic, multiScatteringMetallic );
	#else
		computeMultiscattering( material.dfg, material.specularColor, material.specularF90, singleScatteringDielectric, multiScatteringDielectric );
		computeMultiscattering( material.dfg, material.diffuseColor, material.specularF90, singleScatteringMetallic, multiScatteringMetallic );
	#endif
	vec3 singleScattering = mix( singleScatteringDielectric, singleScatteringMetallic, material.metalness );
	vec3 multiScattering = mix( multiScatteringDielectric, multiScatteringMetallic, material.metalness );
	vec3 totalScatteringDielectric = singleScatteringDielectric + multiScatteringDielectric;
	#ifdef USE_DIFFUSE_ROUGHNESS
		float dotNV = saturate( dot( geometryNormal, geometryViewDir ) );
		vec3 diffuse = EON_DirectionalAlbedo( material.diffuseColor, material.diffuseRoughness, dotNV ) * ( 1.0 - material.metalness ) * ( 1.0 - totalScatteringDielectric );
	#else
		vec3 diffuse = material.diffuseContribution * ( 1.0 - totalScatteringDielectric );
	#endif
	vec3 cosineWeightedIrradiance = irradiance * RECIPROCAL_PI;
	vec3 indirectSpecular = radiance * singleScattering;
	indirectSpecular += multiScattering * cosineWeightedIrradiance;
	vec3 indirectDiffuse = diffuse * cosineWeightedIrradiance;
	#ifdef USE_SHEEN
		float sheenAlbedo = IBLSheenBRDF( geometryNormal, geometryViewDir, material.sheenRoughness );
		float sheenEnergyComp = 1.0 - max3( material.sheenColor ) * sheenAlbedo;
		indirectSpecular *= sheenEnergyComp;
		indirectDiffuse *= sheenEnergyComp;
	#endif
	reflectedLight.indirectSpecular += indirectSpecular;
	reflectedLight.indirectDiffuse += indirectDiffuse;
}
#define RE_Direct				RE_Direct_Physical
#define RE_Direct_RectArea		RE_Direct_RectArea_Physical
#define RE_IndirectDiffuse		RE_IndirectDiffuse_Physical
#define RE_IndirectSpecular		RE_IndirectSpecular_Physical
float computeSpecularOcclusion( const in float dotNV, const in float ambientOcclusion, const in float roughness ) {
	return saturate( pow( dotNV + ambientOcclusion, exp2( - 16.0 * roughness - 1.0 ) ) - 1.0 + ambientOcclusion );
}`,Cm=`
vec3 geometryPosition = - vViewPosition;
vec3 geometryNormal = normal;
vec3 geometryViewDir = ( isOrthographic ) ? vec3( 0, 0, 1 ) : normalize( vViewPosition );
vec3 geometryClearcoatNormal = vec3( 0.0 );
#ifdef USE_CLEARCOAT
	geometryClearcoatNormal = clearcoatNormal;
#endif
#ifdef USE_IRIDESCENCE
	float dotNVi = saturate( dot( normal, geometryViewDir ) );
	if ( material.iridescenceThickness == 0.0 ) {
		material.iridescence = 0.0;
	} else {
		material.iridescence = saturate( material.iridescence );
	}
	if ( material.iridescence > 0.0 ) {
		vec3 iridescenceFresnelDielectric = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.specularColor );
		vec3 iridescenceFresnelMetallic = evalIridescence( 1.0, material.iridescenceIOR, dotNVi, material.iridescenceThickness, material.diffuseColor );
		material.iridescenceFresnel = mix( iridescenceFresnelDielectric, iridescenceFresnelMetallic, material.metalness );
		material.iridescenceF0Dielectric = Schlick_to_F0( iridescenceFresnelDielectric, 1.0, dotNVi );
		material.iridescenceF0Metallic = Schlick_to_F0( iridescenceFresnelMetallic, 1.0, dotNVi );
	}
#endif
#ifdef STANDARD
	float dotNVms = saturate( dot( geometryNormal, geometryViewDir ) );
	material.dfg = texture2D( dfgLUT, vec2( material.roughness, dotNVms ) ).rg;
	#if ( NUM_SUN_LIGHTS > 0 || NUM_DIR_LIGHTS > 0 || NUM_POINT_LIGHTS > 0 || NUM_SPOT_LIGHTS > 0 )
		float EssMs = material.dfg.x + material.dfg.y;
		material.multiScatteringCompensation = 1.0 + material.specularColorBlended * ( 1.0 / EssMs - 1.0 );
	#endif
#endif
IncidentLight directLight;
#if ( NUM_POINT_LIGHTS > 0 ) && defined( RE_Direct )
	PointLight pointLight;
	#if defined( USE_SHADOWMAP ) && NUM_POINT_LIGHT_SHADOWS > 0
	PointLightShadow pointLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHTS; i ++ ) {
		pointLight = pointLights[ i ];
		getPointLightInfo( pointLight, geometryPosition, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_POINT_LIGHT_SHADOWS ) && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
		pointLightShadow = pointLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getPointShadow( pointShadowMap[ i ], pointLightShadow.shadowMapSize, pointLightShadow.shadowIntensity, pointLightShadow.shadowBias, pointLightShadow.shadowRadius, vPointShadowCoord[ i ], pointLightShadow.shadowCameraNear, pointLightShadow.shadowCameraFar ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SPOT_LIGHTS > 0 ) && defined( RE_Direct )
	SpotLight spotLight;
	vec4 spotColor;
	vec3 spotLightCoord;
	bool inSpotLightMap;
	#if defined( USE_SHADOWMAP ) && NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHTS; i ++ ) {
		spotLight = spotLights[ i ];
		getSpotLightInfo( spotLight, geometryPosition, directLight );
		#if ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#define SPOT_LIGHT_MAP_INDEX UNROLLED_LOOP_INDEX
		#elif ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		#define SPOT_LIGHT_MAP_INDEX NUM_SPOT_LIGHT_MAPS
		#else
		#define SPOT_LIGHT_MAP_INDEX ( UNROLLED_LOOP_INDEX - NUM_SPOT_LIGHT_SHADOWS + NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS )
		#endif
		#if ( SPOT_LIGHT_MAP_INDEX < NUM_SPOT_LIGHT_MAPS )
			spotLightCoord = vSpotLightCoord[ i ].xyz / vSpotLightCoord[ i ].w;
			inSpotLightMap = all( lessThan( abs( spotLightCoord * 2. - 1. ), vec3( 1.0 ) ) );
			spotColor = texture2D( spotLightMap[ SPOT_LIGHT_MAP_INDEX ], spotLightCoord.xy );
			directLight.color = inSpotLightMap ? directLight.color * spotColor.rgb : directLight.color;
		#endif
		#undef SPOT_LIGHT_MAP_INDEX
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
		spotLightShadow = spotLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( spotShadowMap[ i ], spotLightShadow.shadowMapSize, spotLightShadow.shadowIntensity, spotLightShadow.shadowBias, spotLightShadow.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_SUN_LIGHTS > 0 ) && defined( RE_Direct )
	SunLight sunLight;
	#if defined( USE_SHADOWMAP ) && NUM_SUN_LIGHT_SHADOWS > 0
	SunLightShadow sunLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SUN_LIGHTS; i ++ ) {
		sunLight = sunLights[ i ];
		getSunLightInfo( sunLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_SUN_LIGHT_SHADOWS )
		sunLightShadow = sunLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getSunShadow( sunShadowMap[ i ], sunLightShadow, UNROLLED_LOOP_INDEX ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_DIR_LIGHTS > 0 ) && defined( RE_Direct )
	DirectionalLight directionalLight;
	#if defined( USE_SHADOWMAP ) && NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLightShadow;
	#endif
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHTS; i ++ ) {
		directionalLight = directionalLights[ i ];
		getDirectionalLightInfo( directionalLight, directLight );
		#if defined( USE_SHADOWMAP ) && ( UNROLLED_LOOP_INDEX < NUM_DIR_LIGHT_SHADOWS )
		directionalLightShadow = directionalLightShadows[ i ];
		directLight.color *= ( directLight.visible && receiveShadow ) ? getShadow( directionalShadowMap[ i ], directionalLightShadow.shadowMapSize, directionalLightShadow.shadowIntensity, directionalLightShadow.shadowBias, directionalLightShadow.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
		#endif
		RE_Direct( directLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if ( NUM_RECT_AREA_LIGHTS > 0 ) && defined( RE_Direct_RectArea )
	RectAreaLight rectAreaLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_RECT_AREA_LIGHTS; i ++ ) {
		rectAreaLight = rectAreaLights[ i ];
		RE_Direct_RectArea( rectAreaLight, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
	}
	#pragma unroll_loop_end
#endif
#if defined( RE_IndirectDiffuse )
	vec3 iblIrradiance = vec3( 0.0 );
	vec3 irradiance = getAmbientLightIrradiance( ambientLightColor );
	#if defined( USE_LIGHT_PROBES )
		irradiance += getLightProbeIrradiance( lightProbe, geometryNormal );
	#endif
	#if ( NUM_HEMI_LIGHTS > 0 )
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_HEMI_LIGHTS; i ++ ) {
			irradiance += getHemisphereLightIrradiance( hemisphereLights[ i ], geometryNormal );
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_LIGHT_PROBE_GRIDS > 0
		vec3 probeWorldPos = ( ( vec4( geometryPosition, 1.0 ) - viewMatrix[ 3 ] ) * viewMatrix ).xyz;
		vec3 probeWorldNormal = transformNormalByInverseViewMatrix( geometryNormal, viewMatrix );
		int probeGrid = - 1;
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_LIGHT_PROBE_GRIDS; i ++ ) {
			if ( isInsideLightProbeGrid( probeWorldPos, probeWorldNormal, probesMin[ i ], probesMax[ i ], probesResolution[ i ] ) ) probeGrid = UNROLLED_LOOP_INDEX;
		}
		#pragma unroll_loop_end
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_LIGHT_PROBE_GRIDS; i ++ ) {
			if ( probeGrid == UNROLLED_LOOP_INDEX ) irradiance += getLightProbeGridIrradiance( probesSH[ i ], probesMin[ i ], probesMax[ i ], probesResolution[ i ], probeWorldPos, probeWorldNormal );
		}
		#pragma unroll_loop_end
	#endif
#endif
#if defined( RE_IndirectSpecular )
	vec3 radiance = vec3( 0.0 );
	vec3 clearcoatRadiance = vec3( 0.0 );
#endif`,Rm=`#if defined( RE_IndirectDiffuse )
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		vec3 lightMapIrradiance = lightMapTexel.rgb * lightMapIntensity;
		irradiance += lightMapIrradiance;
	#endif
	#if defined( USE_ENVMAP ) && defined( ENVMAP_TYPE_PMREM )
		#if defined( STANDARD ) || defined( LAMBERT ) || defined( PHONG )
			iblIrradiance += getIBLIrradiance( geometryNormal );
		#endif
	#endif
#endif
#if defined( USE_ENVMAP ) && defined( RE_IndirectSpecular )
	#ifdef USE_ANISOTROPY
		vec3 iblRadiance = getIBLAnisotropyRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
	#else
		vec3 iblRadiance = getIBLRadiance( geometryViewDir, geometryNormal, material.roughness );
	#endif
	#ifdef USE_RETROREFLECTION
		#ifdef USE_ANISOTROPY
			vec3 retroIBLRadiance = getIBLAnisotropyRetroRadiance( geometryViewDir, geometryNormal, material.roughness, material.anisotropyB, material.anisotropy );
		#else
			vec3 retroIBLRadiance = getIBLRetroRadiance( geometryViewDir, geometryNormal, material.roughness );
		#endif
		iblRadiance = mix( iblRadiance, retroIBLRadiance, saturate( material.retroreflectivity ) );
	#endif
	radiance += iblRadiance;
	#ifdef USE_CLEARCOAT
		clearcoatRadiance += getIBLRadiance( geometryViewDir, geometryClearcoatNormal, material.clearcoatRoughness );
	#endif
#endif`,Pm=`#if defined( RE_IndirectDiffuse )
	#if defined( LAMBERT ) || defined( PHONG )
		irradiance += iblIrradiance;
	#endif
	RE_IndirectDiffuse( irradiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif
#if defined( RE_IndirectSpecular )
	RE_IndirectSpecular( radiance, iblIrradiance, clearcoatRadiance, geometryPosition, geometryNormal, geometryViewDir, geometryClearcoatNormal, material, reflectedLight );
#endif`,Im=`#if NUM_LIGHT_PROBE_GRIDS > 0
uniform highp sampler3D probesSH[ NUM_LIGHT_PROBE_GRIDS ];
uniform vec3 probesMin[ NUM_LIGHT_PROBE_GRIDS ];
uniform vec3 probesMax[ NUM_LIGHT_PROBE_GRIDS ];
uniform vec3 probesResolution[ NUM_LIGHT_PROBE_GRIDS ];
bool isInsideLightProbeGrid( vec3 worldPos, vec3 worldNormal, vec3 probesMin, vec3 probesMax, vec3 probesResolution ) {
	vec3 spacing = ( probesMax - probesMin ) / probesResolution;
	vec3 samplePos = worldPos + worldNormal * spacing * 0.5;
	vec3 margin = spacing * 0.01;
	return all( greaterThanEqual( samplePos, probesMin - margin ) ) && all( lessThanEqual( samplePos, probesMax + margin ) );
}
vec3 getLightProbeGridIrradiance( highp sampler3D probesSH, vec3 probesMin, vec3 probesMax, vec3 probesResolution, vec3 worldPos, vec3 worldNormal ) {
	vec3 res = probesResolution;
	vec3 gridRange = probesMax - probesMin;
	vec3 probeSpacing = gridRange / res;
	vec3 samplePos = worldPos + worldNormal * probeSpacing * 0.5;
	vec3 uvw = clamp( ( samplePos - probesMin ) / gridRange, 0.5 / res, 1.0 - 0.5 / res );
	float nz          = res.z;
	float paddedSlices = nz + 2.0;
	float atlasDepth  = 7.0 * paddedSlices;
	float uvZBase     = uvw.z * nz + 1.0;
	vec4 s0 = texture( probesSH, vec3( uvw.xy, ( uvZBase                       ) / atlasDepth ) );
	vec4 s1 = texture( probesSH, vec3( uvw.xy, ( uvZBase +       paddedSlices   ) / atlasDepth ) );
	vec4 s2 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 2.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s3 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 3.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s4 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 4.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s5 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 5.0 * paddedSlices   ) / atlasDepth ) );
	vec4 s6 = texture( probesSH, vec3( uvw.xy, ( uvZBase + 6.0 * paddedSlices   ) / atlasDepth ) );
	vec3 c0 = s0.xyz;
	vec3 c1 = vec3( s0.w, s1.xy );
	vec3 c2 = vec3( s1.zw, s2.x );
	vec3 c3 = s2.yzw;
	vec3 c4 = s3.xyz;
	vec3 c5 = vec3( s3.w, s4.xy );
	vec3 c6 = vec3( s4.zw, s5.x );
	vec3 c7 = s5.yzw;
	vec3 c8 = s6.xyz;
	float x = worldNormal.x, y = worldNormal.y, z = worldNormal.z;
	vec3 result = c0 * 0.886227;
	result += c1 * 2.0 * 0.511664 * y;
	result += c2 * 2.0 * 0.511664 * z;
	result += c3 * 2.0 * 0.511664 * x;
	result += c4 * 2.0 * 0.429043 * x * y;
	result += c5 * 2.0 * 0.429043 * y * z;
	result += c6 * ( 0.743125 * z * z - 0.247708 );
	result += c7 * 2.0 * 0.429043 * x * z;
	result += c8 * 0.429043 * ( x * x - y * y );
	return max( result, vec3( 0.0 ) );
}
#endif`,Lm=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	gl_FragDepth = vIsPerspective == 0.0 ? gl_FragCoord.z : log2( vFragDepth ) * logDepthBufFC * 0.5;
#endif`,Dm=`#if defined( USE_LOGARITHMIC_DEPTH_BUFFER )
	uniform float logDepthBufFC;
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Nm=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	varying float vFragDepth;
	varying float vIsPerspective;
#endif`,Um=`#ifdef USE_LOGARITHMIC_DEPTH_BUFFER
	vFragDepth = 1.0 + gl_Position.w;
	vIsPerspective = float( isPerspectiveMatrix( projectionMatrix ) );
#endif`,Fm=`#ifdef USE_MAP
	vec4 sampledDiffuseColor = texture2D( map, vMapUv );
	#ifdef DECODE_VIDEO_TEXTURE
		sampledDiffuseColor = sRGBTransferEOTF( sampledDiffuseColor );
	#endif
	diffuseColor *= sampledDiffuseColor;
#endif`,Om=`#ifdef USE_MAP
	uniform sampler2D map;
#endif`,Bm=`#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
	#if defined( USE_POINTS_UV )
		vec2 uv = vUv;
	#else
		vec2 uv = ( uvTransform * vec3( gl_PointCoord.x, 1.0 - gl_PointCoord.y, 1 ) ).xy;
	#endif
#endif
#ifdef USE_MAP
	diffuseColor *= texture2D( map, uv );
#endif
#ifdef USE_ALPHAMAP
	diffuseColor.a *= texture2D( alphaMap, uv ).g;
#endif`,km=`#if defined( USE_POINTS_UV )
	varying vec2 vUv;
#else
	#if defined( USE_MAP ) || defined( USE_ALPHAMAP )
		uniform mat3 uvTransform;
	#endif
#endif
#ifdef USE_MAP
	uniform sampler2D map;
#endif
#ifdef USE_ALPHAMAP
	uniform sampler2D alphaMap;
#endif`,zm=`float metalnessFactor = metalness;
#ifdef USE_METALNESSMAP
	vec4 texelMetalness = texture2D( metalnessMap, vMetalnessMapUv );
	metalnessFactor *= texelMetalness.b;
#endif`,Gm=`#ifdef USE_METALNESSMAP
	uniform sampler2D metalnessMap;
#endif`,Vm=`#ifdef USE_INSTANCING_MORPH
	float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	float morphTargetBaseInfluence = texelFetch( morphTexture, ivec2( 0, gl_InstanceID ), 0 ).r;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		morphTargetInfluences[i] =  texelFetch( morphTexture, ivec2( i + 1, gl_InstanceID ), 0 ).r;
	}
#endif`,Hm=`#if defined( USE_MORPHCOLORS )
	#if defined( USE_COLOR_ALPHA )
		vColor *= morphTargetBaseInfluence;
	#elif defined( USE_COLOR )
		vColor.rgb *= morphTargetBaseInfluence;
	#endif
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		#if defined( USE_COLOR_ALPHA )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor += getMorph( gl_VertexID, i, 2 ) * morphTargetInfluences[ i ];
		#elif defined( USE_COLOR )
			if ( morphTargetInfluences[ i ] != 0.0 ) vColor.rgb += getMorph( gl_VertexID, i, 2 ).rgb * morphTargetInfluences[ i ];
		#endif
	}
#endif`,Wm=`#ifdef USE_MORPHNORMALS
	objectNormal *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) objectNormal += getMorph( gl_VertexID, i, 1 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Xm=`#ifdef USE_MORPHTARGETS
	#ifndef USE_INSTANCING_MORPH
		uniform float morphTargetBaseInfluence;
		uniform float morphTargetInfluences[ MORPHTARGETS_COUNT ];
	#endif
	uniform sampler2DArray morphTargetsTexture;
	uniform ivec2 morphTargetsTextureSize;
	vec4 getMorph( const in int vertexIndex, const in int morphTargetIndex, const in int offset ) {
		int texelIndex = vertexIndex * MORPHTARGETS_TEXTURE_STRIDE + offset;
		int y = texelIndex / morphTargetsTextureSize.x;
		int x = texelIndex - y * morphTargetsTextureSize.x;
		ivec3 morphUV = ivec3( x, y, morphTargetIndex );
		return texelFetch( morphTargetsTexture, morphUV, 0 );
	}
#endif`,qm=`#ifdef USE_MORPHTARGETS
	transformed *= morphTargetBaseInfluence;
	for ( int i = 0; i < MORPHTARGETS_COUNT; i ++ ) {
		if ( morphTargetInfluences[ i ] != 0.0 ) transformed += getMorph( gl_VertexID, i, 0 ).xyz * morphTargetInfluences[ i ];
	}
#endif`,Ym=`float faceDirection = gl_FrontFacing ? 1.0 : - 1.0;
#ifdef FLAT_SHADED
	vec3 fdx = dFdx( vViewPosition );
	vec3 fdy = dFdy( vViewPosition );
	vec3 normal = normalize( cross( fdx, fdy ) );
#else
	vec3 normal = normalize( vNormal );
	#ifdef DOUBLE_SIDED
		normal *= faceDirection;
	#endif
#endif
#if defined( USE_NORMALMAP_TANGENTSPACE ) || defined( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY )
	#ifdef USE_TANGENT
		mat3 tbn = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn = getTangentFrame( - vViewPosition, normal,
		#if defined( USE_NORMALMAP )
			vNormalMapUv
		#elif defined( USE_CLEARCOAT_NORMALMAP )
			vClearcoatNormalMapUv
		#else
			vUv
		#endif
		);
	#endif
	#ifdef DOUBLE_SIDED
		tbn[0] *= faceDirection;
		tbn[1] *= faceDirection;
	#endif
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	#ifdef USE_TANGENT
		mat3 tbn2 = mat3( normalize( vTangent ), normalize( vBitangent ), normal );
	#else
		mat3 tbn2 = getTangentFrame( - vViewPosition, normal, vClearcoatNormalMapUv );
	#endif
	#ifdef DOUBLE_SIDED
		tbn2[0] *= faceDirection;
		tbn2[1] *= faceDirection;
	#endif
#endif
vec3 nonPerturbedNormal = normal;`,$m=`#ifdef USE_NORMALMAP_OBJECTSPACE
	normal = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#ifdef FLIP_SIDED
		normal = - normal;
	#endif
	#ifdef DOUBLE_SIDED
		normal = normal * faceDirection;
	#endif
	normal = normalize( normalMatrix * normal );
#elif defined( USE_NORMALMAP_TANGENTSPACE )
	vec3 mapN = texture2D( normalMap, vNormalMapUv ).xyz * 2.0 - 1.0;
	#if defined( USE_PACKED_NORMALMAP )
		mapN = vec3( mapN.xy, sqrt( saturate( 1.0 - dot( mapN.xy, mapN.xy ) ) ) );
	#endif
	mapN.xy *= normalScale;
	normal = normalize( tbn * mapN );
#elif defined( USE_BUMPMAP )
	normal = perturbNormalArb( - vViewPosition, normal, dHdxy_fwd(), faceDirection );
#endif`,Zm=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Jm=`#ifndef FLAT_SHADED
	varying vec3 vNormal;
	#ifdef USE_TANGENT
		varying vec3 vTangent;
		varying vec3 vBitangent;
	#endif
#endif`,Km=`#ifndef FLAT_SHADED
	vNormal = normalize( transformedNormal );
	#ifdef USE_TANGENT
		vTangent = normalize( transformedTangent );
		vBitangent = normalize( cross( vNormal, vTangent ) * tangent.w );
		#ifdef FLIP_SIDED
			vBitangent = - vBitangent;
		#endif
	#endif
#endif`,jm=`#ifdef USE_NORMALMAP
	uniform sampler2D normalMap;
	uniform vec2 normalScale;
#endif
#ifdef USE_NORMALMAP_OBJECTSPACE
	uniform mat3 normalMatrix;
#endif
#if ! defined ( USE_TANGENT ) && ( defined ( USE_NORMALMAP_TANGENTSPACE ) || defined ( USE_CLEARCOAT_NORMALMAP ) || defined( USE_ANISOTROPY ) )
	mat3 getTangentFrame( vec3 eye_pos, vec3 surf_norm, vec2 uv ) {
		vec3 q0 = dFdx( eye_pos.xyz );
		vec3 q1 = dFdy( eye_pos.xyz );
		vec2 st0 = dFdx( uv.st );
		vec2 st1 = dFdy( uv.st );
		vec3 N = surf_norm;
		vec3 q1perp = cross( q1, N );
		vec3 q0perp = cross( N, q0 );
		vec3 T = q1perp * st0.x + q0perp * st1.x;
		vec3 B = q1perp * st0.y + q0perp * st1.y;
		float det = max( dot( T, T ), dot( B, B ) );
		float scale = ( det == 0.0 ) ? 0.0 : inversesqrt( det );
		return mat3( T * scale, B * scale, N );
	}
#endif`,Qm=`#ifdef USE_CLEARCOAT
	vec3 clearcoatNormal = nonPerturbedNormal;
#endif`,tg=`#ifdef USE_CLEARCOAT_NORMALMAP
	vec3 clearcoatMapN = texture2D( clearcoatNormalMap, vClearcoatNormalMapUv ).xyz * 2.0 - 1.0;
	clearcoatMapN.xy *= clearcoatNormalScale;
	clearcoatNormal = normalize( tbn2 * clearcoatMapN );
#endif`,eg=`#ifdef USE_CLEARCOATMAP
	uniform sampler2D clearcoatMap;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform sampler2D clearcoatNormalMap;
	uniform vec2 clearcoatNormalScale;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform sampler2D clearcoatRoughnessMap;
#endif`,ng=`#ifdef USE_IRIDESCENCEMAP
	uniform sampler2D iridescenceMap;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform sampler2D iridescenceThicknessMap;
#endif`,ig=`#ifdef OPAQUE
diffuseColor.a = 1.0;
#endif
#ifdef USE_TRANSMISSION
diffuseColor.a *= material.transmissionAlpha;
#endif
gl_FragColor = vec4( outgoingLight, diffuseColor.a );`,sg=`vec3 packNormalToRGB( const in vec3 normal ) {
	return normalize( normal ) * 0.5 + 0.5;
}
vec3 unpackRGBToNormal( const in vec3 rgb ) {
	return 2.0 * rgb.xyz - 1.0;
}
const float PackUpscale = 256. / 255.;const float UnpackDownscale = 255. / 256.;const float ShiftRight8 = 1. / 256.;
const float Inv255 = 1. / 255.;
const vec4 PackFactors = vec4( 1.0, 256.0, 256.0 * 256.0, 256.0 * 256.0 * 256.0 );
const vec2 UnpackFactors2 = vec2( UnpackDownscale, 1.0 / PackFactors.g );
const vec3 UnpackFactors3 = vec3( UnpackDownscale / PackFactors.rg, 1.0 / PackFactors.b );
const vec4 UnpackFactors4 = vec4( UnpackDownscale / PackFactors.rgb, 1.0 / PackFactors.a );
vec4 packDepthToRGBA( const in float v ) {
	if( v <= 0.0 )
		return vec4( 0., 0., 0., 0. );
	if( v >= 1.0 )
		return vec4( 1., 1., 1., 1. );
	float vuf;
	float af = modf( v * PackFactors.a, vuf );
	float bf = modf( vuf * ShiftRight8, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec4( vuf * Inv255, gf * PackUpscale, bf * PackUpscale, af );
}
vec3 packDepthToRGB( const in float v ) {
	if( v <= 0.0 )
		return vec3( 0., 0., 0. );
	if( v >= 1.0 )
		return vec3( 1., 1., 1. );
	float vuf;
	float bf = modf( v * PackFactors.b, vuf );
	float gf = modf( vuf * ShiftRight8, vuf );
	return vec3( vuf * Inv255, gf * PackUpscale, bf );
}
vec2 packDepthToRG( const in float v ) {
	if( v <= 0.0 )
		return vec2( 0., 0. );
	if( v >= 1.0 )
		return vec2( 1., 1. );
	float vuf;
	float gf = modf( v * 256., vuf );
	return vec2( vuf * Inv255, gf );
}
float unpackRGBAToDepth( const in vec4 v ) {
	return dot( v, UnpackFactors4 );
}
float unpackRGBToDepth( const in vec3 v ) {
	return dot( v, UnpackFactors3 );
}
float unpackRGToDepth( const in vec2 v ) {
	return v.r * UnpackFactors2.r + v.g * UnpackFactors2.g;
}
vec4 pack2HalfToRGBA( const in vec2 v ) {
	vec4 r = vec4( v.x, fract( v.x * 255.0 ), v.y, fract( v.y * 255.0 ) );
	return vec4( r.x - r.y / 255.0, r.y, r.z - r.w / 255.0, r.w );
}
vec2 unpackRGBATo2Half( const in vec4 v ) {
	return vec2( v.x + ( v.y / 255.0 ), v.z + ( v.w / 255.0 ) );
}
float viewZToOrthographicDepth( const in float viewZ, const in float near, const in float far ) {
	return ( viewZ + near ) / ( near - far );
}
float orthographicDepthToViewZ( const in float depth, const in float near, const in float far ) {
	#ifdef USE_REVERSED_DEPTH_BUFFER
	
		return depth * ( far - near ) - far;
	#else
		return depth * ( near - far ) - near;
	#endif
}
float viewZToPerspectiveDepth( const in float viewZ, const in float near, const in float far ) {
	return ( ( near + viewZ ) * far ) / ( ( far - near ) * viewZ );
}
float perspectiveDepthToViewZ( const in float depth, const in float near, const in float far ) {
	
	#ifdef USE_REVERSED_DEPTH_BUFFER
		return ( near * far ) / ( ( near - far ) * depth - near );
	#else
		return ( near * far ) / ( ( far - near ) * depth - far );
	#endif
}`,rg=`#ifdef PREMULTIPLIED_ALPHA
	gl_FragColor.rgb *= gl_FragColor.a;
#endif`,ag=`vec4 mvPosition = vec4( transformed, 1.0 );
#ifdef USE_BATCHING
	mvPosition = batchingMatrix * mvPosition;
#endif
#ifdef USE_INSTANCING
	mvPosition = instanceMatrix * mvPosition;
#endif
mvPosition = modelViewMatrix * mvPosition;
gl_Position = projectionMatrix * mvPosition;`,og=`#ifdef DITHERING
	gl_FragColor.rgb = dithering( gl_FragColor.rgb );
#endif`,lg=`#ifdef DITHERING
	vec3 dithering( vec3 color ) {
		float grid_position = rand( gl_FragCoord.xy );
		vec3 dither_shift_RGB = vec3( 0.25 / 255.0, -0.25 / 255.0, 0.25 / 255.0 );
		dither_shift_RGB = mix( 2.0 * dither_shift_RGB, -2.0 * dither_shift_RGB, grid_position );
		return color + dither_shift_RGB;
	}
#endif`,cg=`float roughnessFactor = roughness;
#ifdef USE_ROUGHNESSMAP
	vec4 texelRoughness = texture2D( roughnessMap, vRoughnessMapUv );
	roughnessFactor *= texelRoughness.g;
#endif`,hg=`#ifdef USE_ROUGHNESSMAP
	uniform sampler2D roughnessMap;
#endif`,ug=`#if NUM_SPOT_LIGHT_COORDS > 0
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#if NUM_SPOT_LIGHT_MAPS > 0
	uniform sampler2D spotLightMap[ NUM_SPOT_LIGHT_MAPS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_SUN_LIGHT_SHADOWS > 0
		#define SUN_LIGHT_CASCADES 2
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow sunShadowMap[ NUM_SUN_LIGHT_SHADOWS ];
		#else
			uniform sampler2D sunShadowMap[ NUM_SUN_LIGHT_SHADOWS ];
		#endif
		uniform mat4 sunShadowMatrix[ NUM_SUN_LIGHT_SHADOWS * SUN_LIGHT_CASCADES ];
		uniform vec4 sunShadowCascade[ NUM_SUN_LIGHT_SHADOWS * SUN_LIGHT_CASCADES ];
		varying vec4 vSunShadowWorldPosition;
		varying vec3 vSunShadowWorldNormal;
		struct SunLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SunLightShadow sunLightShadows[ NUM_SUN_LIGHT_SHADOWS ];
	#endif
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#else
			uniform sampler2D directionalShadowMap[ NUM_DIR_LIGHT_SHADOWS ];
		#endif
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform sampler2DShadow spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#else
			uniform sampler2D spotShadowMap[ NUM_SPOT_LIGHT_SHADOWS ];
		#endif
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#if defined( SHADOWMAP_TYPE_PCF )
			uniform samplerCubeShadow pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#elif defined( SHADOWMAP_TYPE_BASIC )
			uniform samplerCube pointShadowMap[ NUM_POINT_LIGHT_SHADOWS ];
		#endif
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float interleavedGradientNoise( vec2 position ) {
			return fract( 52.9829189 * fract( dot( position, vec2( 0.06711056, 0.00583715 ) ) ) );
		}
		vec2 vogelDiskSample( int sampleIndex, int samplesCount, float phi ) {
			const float goldenAngle = 2.399963229728653;
			float r = sqrt( ( float( sampleIndex ) + 0.5 ) / float( samplesCount ) );
			float theta = float( sampleIndex ) * goldenAngle + phi;
			return vec2( cos( theta ), sin( theta ) ) * r;
		}
	#endif
	#if defined( SHADOWMAP_TYPE_PCF )
		float getShadow( sampler2DShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 texelSize = vec2( 1.0 ) / shadowMapSize;
				vec2 radius = shadowRadius * texelSize;
				float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
				shadow = (
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 0, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 1, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 2, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 3, 5, phi ) * radius, shadowCoord.z ) ) +
					texture( shadowMap, vec3( shadowCoord.xy + vogelDiskSample( 4, 5, phi ) * radius, shadowCoord.z ) )
				) * 0.2;
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#elif defined( SHADOWMAP_TYPE_VSM )
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				vec2 distribution = texture2D( shadowMap, shadowCoord.xy ).rg;
				float mean = distribution.x;
				float variance = distribution.y * distribution.y;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					float hard_shadow = step( mean, shadowCoord.z );
				#else
					float hard_shadow = step( shadowCoord.z, mean );
				#endif
				
				if ( hard_shadow == 1.0 ) {
					shadow = 1.0;
				} else {
					variance = max( variance, 0.0000001 );
					float d = shadowCoord.z - mean;
					float p_max = variance / ( variance + d * d );
					p_max = clamp( ( p_max - 0.3 ) / 0.65, 0.0, 1.0 );
					shadow = max( hard_shadow, p_max );
				}
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#else
		float getShadow( sampler2D shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord ) {
			float shadow = 1.0;
			shadowCoord.xyz /= shadowCoord.w;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				shadowCoord.z -= shadowBias;
			#else
				shadowCoord.z += shadowBias;
			#endif
			bool inFrustum = shadowCoord.x >= 0.0 && shadowCoord.x <= 1.0 && shadowCoord.y >= 0.0 && shadowCoord.y <= 1.0;
			bool frustumTest = inFrustum && shadowCoord.z <= 1.0;
			if ( frustumTest ) {
				float depth = texture2D( shadowMap, shadowCoord.xy ).r;
				#ifdef USE_REVERSED_DEPTH_BUFFER
					shadow = step( depth, shadowCoord.z );
				#else
					shadow = step( shadowCoord.z, depth );
				#endif
			}
			return mix( 1.0, shadow, shadowIntensity );
		}
	#endif
	#if NUM_SUN_LIGHT_SHADOWS > 0
		float getSunShadow(
			#if defined( SHADOWMAP_TYPE_PCF )
				sampler2DShadow shadowMap,
			#else
				sampler2D shadowMap,
			#endif
			SunLightShadow sunLightShadow,
			int shadowIndex
		) {
			vec4 shadowWorldPosition = vec4( vSunShadowWorldPosition.xyz + vSunShadowWorldNormal * sunLightShadow.shadowNormalBias, 1.0 );
			float viewDepth = vSunShadowWorldPosition.w;
			int cascadeOffset = shadowIndex * SUN_LIGHT_CASCADES;
			float shadow = 1.0;
			for ( int i = SUN_LIGHT_CASCADES - 1; i >= 0; i -- ) {
				vec4 cascade = sunShadowCascade[ cascadeOffset + i ];
				if ( viewDepth >= cascade.x && viewDepth < cascade.y ) {
					float cascadeShadow = getShadow(
						shadowMap,
						sunLightShadow.shadowMapSize,
						sunLightShadow.shadowIntensity,
						sunLightShadow.shadowBias,
						sunLightShadow.shadowRadius,
						sunShadowMatrix[ cascadeOffset + i ] * shadowWorldPosition
					);
					shadow = mix( cascadeShadow, shadow, smoothstep( cascade.z, cascade.y, viewDepth ) );
				}
			}
			return shadow;
		}
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
	#if defined( SHADOWMAP_TYPE_PCF )
	float getPointShadow( samplerCubeShadow shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 bd3D = normalize( lightToPosition );
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			float lightDistance = length( lightToPosition );
			float depthRange = shadowCameraFar - shadowCameraNear;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				float depthA = - shadowCameraNear / depthRange - shadowBias;
				float depthB = shadowCameraNear * shadowCameraFar / ( depthRange * lightDistance );
			#else
				float depthA = shadowCameraFar / depthRange + shadowBias;
				float depthB = - shadowCameraNear * shadowCameraFar / ( depthRange * lightDistance );
			#endif
			float texelSize = shadowRadius / shadowMapSize.x;
			vec3 absDir = abs( bd3D );
			vec3 tangent = absDir.x > absDir.z ? vec3( 0.0, 1.0, 0.0 ) : vec3( 1.0, 0.0, 0.0 );
			tangent = normalize( cross( bd3D, tangent ) );
			vec3 bitangent = cross( bd3D, tangent );
			float phi = interleavedGradientNoise( gl_FragCoord.xy ) * PI2;
			shadow = 0.0;
			for ( int i = 0; i < 5; i ++ ) {
				vec2 offset = vogelDiskSample( i, 5, phi );
				vec3 direction = bd3D + ( tangent * offset.x + bitangent * offset.y ) * texelSize;
				vec3 absDirection = abs( direction );
				float dp = depthA + depthB / max( max( absDirection.x, absDirection.y ), absDirection.z );
				shadow += texture( shadowMap, vec4( direction, dp ) );
			}
			shadow *= 0.2;
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#elif defined( SHADOWMAP_TYPE_BASIC )
	float getPointShadow( samplerCube shadowMap, vec2 shadowMapSize, float shadowIntensity, float shadowBias, float shadowRadius, vec4 shadowCoord, float shadowCameraNear, float shadowCameraFar ) {
		float shadow = 1.0;
		vec3 lightToPosition = shadowCoord.xyz;
		vec3 absVec = abs( lightToPosition );
		float viewSpaceZ = max( max( absVec.x, absVec.y ), absVec.z );
		if ( viewSpaceZ - shadowCameraFar <= 0.0 && viewSpaceZ - shadowCameraNear >= 0.0 ) {
			float dp = ( shadowCameraFar * ( viewSpaceZ - shadowCameraNear ) ) / ( viewSpaceZ * ( shadowCameraFar - shadowCameraNear ) );
			dp += shadowBias;
			vec3 bd3D = normalize( lightToPosition );
			float depth = textureCube( shadowMap, bd3D ).r;
			#ifdef USE_REVERSED_DEPTH_BUFFER
				depth = 1.0 - depth;
			#endif
			shadow = step( dp, depth );
		}
		return mix( 1.0, shadow, shadowIntensity );
	}
	#endif
	#endif
#endif`,dg=`#if NUM_SPOT_LIGHT_COORDS > 0
	uniform mat4 spotLightMatrix[ NUM_SPOT_LIGHT_COORDS ];
	varying vec4 vSpotLightCoord[ NUM_SPOT_LIGHT_COORDS ];
#endif
#ifdef USE_SHADOWMAP
	#if NUM_SUN_LIGHT_SHADOWS > 0
		varying vec4 vSunShadowWorldPosition;
		varying vec3 vSunShadowWorldNormal;
	#endif
	#if NUM_DIR_LIGHT_SHADOWS > 0
		uniform mat4 directionalShadowMatrix[ NUM_DIR_LIGHT_SHADOWS ];
		varying vec4 vDirectionalShadowCoord[ NUM_DIR_LIGHT_SHADOWS ];
		struct DirectionalLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform DirectionalLightShadow directionalLightShadows[ NUM_DIR_LIGHT_SHADOWS ];
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
		struct SpotLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
		};
		uniform SpotLightShadow spotLightShadows[ NUM_SPOT_LIGHT_SHADOWS ];
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		uniform mat4 pointShadowMatrix[ NUM_POINT_LIGHT_SHADOWS ];
		varying vec4 vPointShadowCoord[ NUM_POINT_LIGHT_SHADOWS ];
		struct PointLightShadow {
			float shadowIntensity;
			float shadowBias;
			float shadowNormalBias;
			float shadowRadius;
			vec2 shadowMapSize;
			float shadowCameraNear;
			float shadowCameraFar;
		};
		uniform PointLightShadow pointLightShadows[ NUM_POINT_LIGHT_SHADOWS ];
	#endif
#endif`,fg=`#if ( defined( USE_SHADOWMAP ) && ( NUM_DIR_LIGHT_SHADOWS > 0 || NUM_SUN_LIGHT_SHADOWS > 0 || NUM_POINT_LIGHT_SHADOWS > 0 ) ) || ( NUM_SPOT_LIGHT_COORDS > 0 )
	#ifdef HAS_NORMAL
		vec3 shadowWorldNormal = transformNormalByInverseViewMatrix( transformedNormal, viewMatrix );
	#else
		vec3 shadowWorldNormal = vec3( 0.0 );
	#endif
	vec4 shadowWorldPosition;
#endif
#if defined( USE_SHADOWMAP )
	#if NUM_SUN_LIGHT_SHADOWS > 0
		vSunShadowWorldPosition = vec4( worldPosition.xyz, - mvPosition.z );
		vSunShadowWorldNormal = shadowWorldNormal;
	#endif
	#if NUM_DIR_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * directionalLightShadows[ i ].shadowNormalBias, 0 );
			vDirectionalShadowCoord[ i ] = directionalShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0
		#pragma unroll_loop_start
		for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
			shadowWorldPosition = worldPosition + vec4( shadowWorldNormal * pointLightShadows[ i ].shadowNormalBias, 0 );
			vPointShadowCoord[ i ] = pointShadowMatrix[ i ] * shadowWorldPosition;
		}
		#pragma unroll_loop_end
	#endif
#endif
#if NUM_SPOT_LIGHT_COORDS > 0
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_COORDS; i ++ ) {
		shadowWorldPosition = worldPosition;
		#if ( defined( USE_SHADOWMAP ) && UNROLLED_LOOP_INDEX < NUM_SPOT_LIGHT_SHADOWS )
			shadowWorldPosition.xyz += shadowWorldNormal * spotLightShadows[ i ].shadowNormalBias;
		#endif
		vSpotLightCoord[ i ] = spotLightMatrix[ i ] * shadowWorldPosition;
	}
	#pragma unroll_loop_end
#endif`,pg=`float getShadowMask() {
	float shadow = 1.0;
	#ifdef USE_SHADOWMAP
	#if NUM_SUN_LIGHT_SHADOWS > 0
	SunLightShadow sunLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SUN_LIGHT_SHADOWS; i ++ ) {
		sunLight = sunLightShadows[ i ];
		shadow *= receiveShadow ? getSunShadow( sunShadowMap[ i ], sunLight, UNROLLED_LOOP_INDEX ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_DIR_LIGHT_SHADOWS > 0
	DirectionalLightShadow directionalLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_DIR_LIGHT_SHADOWS; i ++ ) {
		directionalLight = directionalLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( directionalShadowMap[ i ], directionalLight.shadowMapSize, directionalLight.shadowIntensity, directionalLight.shadowBias, directionalLight.shadowRadius, vDirectionalShadowCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_SPOT_LIGHT_SHADOWS > 0
	SpotLightShadow spotLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_SPOT_LIGHT_SHADOWS; i ++ ) {
		spotLight = spotLightShadows[ i ];
		shadow *= receiveShadow ? getShadow( spotShadowMap[ i ], spotLight.shadowMapSize, spotLight.shadowIntensity, spotLight.shadowBias, spotLight.shadowRadius, vSpotLightCoord[ i ] ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#if NUM_POINT_LIGHT_SHADOWS > 0 && ( defined( SHADOWMAP_TYPE_PCF ) || defined( SHADOWMAP_TYPE_BASIC ) )
	PointLightShadow pointLight;
	#pragma unroll_loop_start
	for ( int i = 0; i < NUM_POINT_LIGHT_SHADOWS; i ++ ) {
		pointLight = pointLightShadows[ i ];
		shadow *= receiveShadow ? getPointShadow( pointShadowMap[ i ], pointLight.shadowMapSize, pointLight.shadowIntensity, pointLight.shadowBias, pointLight.shadowRadius, vPointShadowCoord[ i ], pointLight.shadowCameraNear, pointLight.shadowCameraFar ) : 1.0;
	}
	#pragma unroll_loop_end
	#endif
	#endif
	return shadow;
}`,mg=`#ifdef USE_SKINNING
	mat4 boneMatX = getBoneMatrix( skinIndex.x );
	mat4 boneMatY = getBoneMatrix( skinIndex.y );
	mat4 boneMatZ = getBoneMatrix( skinIndex.z );
	mat4 boneMatW = getBoneMatrix( skinIndex.w );
#endif`,gg=`#ifdef USE_SKINNING
	uniform mat4 bindMatrix;
	uniform mat4 bindMatrixInverse;
	uniform highp sampler2D boneTexture;
	mat4 getBoneMatrix( const in float i ) {
		int size = textureSize( boneTexture, 0 ).x;
		int j = int( i ) * 4;
		int x = j % size;
		int y = j / size;
		vec4 v1 = texelFetch( boneTexture, ivec2( x, y ), 0 );
		vec4 v2 = texelFetch( boneTexture, ivec2( x + 1, y ), 0 );
		vec4 v3 = texelFetch( boneTexture, ivec2( x + 2, y ), 0 );
		vec4 v4 = texelFetch( boneTexture, ivec2( x + 3, y ), 0 );
		return mat4( v1, v2, v3, v4 );
	}
#endif`,xg=`#ifdef USE_SKINNING
	vec4 skinVertex = bindMatrix * vec4( transformed, 1.0 );
	vec4 skinned = vec4( 0.0 );
	skinned += boneMatX * skinVertex * skinWeight.x;
	skinned += boneMatY * skinVertex * skinWeight.y;
	skinned += boneMatZ * skinVertex * skinWeight.z;
	skinned += boneMatW * skinVertex * skinWeight.w;
	transformed = ( bindMatrixInverse * skinned ).xyz;
#endif`,_g=`#ifdef USE_SKINNING
	mat4 skinMatrix = mat4( 0.0 );
	skinMatrix += skinWeight.x * boneMatX;
	skinMatrix += skinWeight.y * boneMatY;
	skinMatrix += skinWeight.z * boneMatZ;
	skinMatrix += skinWeight.w * boneMatW;
	skinMatrix = bindMatrixInverse * skinMatrix * bindMatrix;
	objectNormal = vec4( skinMatrix * vec4( objectNormal, 0.0 ) ).xyz;
	#ifdef USE_TANGENT
		objectTangent = vec4( skinMatrix * vec4( objectTangent, 0.0 ) ).xyz;
	#endif
#endif`,yg=`float specularStrength;
#ifdef USE_SPECULARMAP
	vec4 texelSpecular = texture2D( specularMap, vSpecularMapUv );
	specularStrength = texelSpecular.r;
#else
	specularStrength = 1.0;
#endif`,vg=`#ifdef USE_SPECULARMAP
	uniform sampler2D specularMap;
#endif`,bg=`#if defined( TONE_MAPPING )
	gl_FragColor.rgb = toneMapping( gl_FragColor.rgb );
#endif`,Mg=`#ifndef saturate
#define saturate( a ) clamp( a, 0.0, 1.0 )
#endif
uniform float toneMappingExposure;
vec3 LinearToneMapping( vec3 color ) {
	return saturate( toneMappingExposure * color );
}
vec3 ReinhardToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	return saturate( color / ( vec3( 1.0 ) + color ) );
}
vec3 CineonToneMapping( vec3 color ) {
	color *= toneMappingExposure;
	color = max( vec3( 0.0 ), color - 0.004 );
	return pow( ( color * ( 6.2 * color + 0.5 ) ) / ( color * ( 6.2 * color + 1.7 ) + 0.06 ), vec3( 2.2 ) );
}
vec3 RRTAndODTFit( vec3 v ) {
	vec3 a = v * ( v + 0.0245786 ) - 0.000090537;
	vec3 b = v * ( 0.983729 * v + 0.4329510 ) + 0.238081;
	return a / b;
}
vec3 ACESFilmicToneMapping( vec3 color ) {
	const mat3 ACESInputMat = mat3(
		vec3( 0.59719, 0.07600, 0.02840 ),		vec3( 0.35458, 0.90834, 0.13383 ),
		vec3( 0.04823, 0.01566, 0.83777 )
	);
	const mat3 ACESOutputMat = mat3(
		vec3(  1.60475, -0.10208, -0.00327 ),		vec3( -0.53108,  1.10813, -0.07276 ),
		vec3( -0.07367, -0.00605,  1.07602 )
	);
	color *= toneMappingExposure / 0.6;
	color = ACESInputMat * color;
	color = RRTAndODTFit( color );
	color = ACESOutputMat * color;
	return saturate( color );
}
const mat3 LINEAR_REC2020_TO_LINEAR_SRGB = mat3(
	vec3( 1.6605, - 0.1246, - 0.0182 ),
	vec3( - 0.5876, 1.1329, - 0.1006 ),
	vec3( - 0.0728, - 0.0083, 1.1187 )
);
const mat3 LINEAR_SRGB_TO_LINEAR_REC2020 = mat3(
	vec3( 0.6274, 0.0691, 0.0164 ),
	vec3( 0.3293, 0.9195, 0.0880 ),
	vec3( 0.0433, 0.0113, 0.8956 )
);
vec3 agxDefaultContrastApprox( vec3 x ) {
	vec3 x2 = x * x;
	vec3 x4 = x2 * x2;
	return + 15.5 * x4 * x2
		- 40.14 * x4 * x
		+ 31.96 * x4
		- 6.868 * x2 * x
		+ 0.4298 * x2
		+ 0.1191 * x
		- 0.00232;
}
vec3 AgXToneMapping( vec3 color ) {
	const mat3 AgXInsetMatrix = mat3(
		vec3( 0.856627153315983, 0.137318972929847, 0.11189821299995 ),
		vec3( 0.0951212405381588, 0.761241990602591, 0.0767994186031903 ),
		vec3( 0.0482516061458583, 0.101439036467562, 0.811302368396859 )
	);
	const mat3 AgXOutsetMatrix = mat3(
		vec3( 1.1271005818144368, - 0.1413297634984383, - 0.14132976349843826 ),
		vec3( - 0.11060664309660323, 1.157823702216272, - 0.11060664309660294 ),
		vec3( - 0.016493938717834573, - 0.016493938717834257, 1.2519364065950405 )
	);
	const float AgxMinEv = - 12.47393;	const float AgxMaxEv = 4.026069;
	color *= toneMappingExposure;
	color = LINEAR_SRGB_TO_LINEAR_REC2020 * color;
	color = AgXInsetMatrix * color;
	color = max( color, 1e-10 );	color = log2( color );
	color = ( color - AgxMinEv ) / ( AgxMaxEv - AgxMinEv );
	color = clamp( color, 0.0, 1.0 );
	color = agxDefaultContrastApprox( color );
	color = AgXOutsetMatrix * color;
	color = pow( max( vec3( 0.0 ), color ), vec3( 2.2 ) );
	color = LINEAR_REC2020_TO_LINEAR_SRGB * color;
	color = clamp( color, 0.0, 1.0 );
	return color;
}
vec3 NeutralToneMapping( vec3 color ) {
	const float StartCompression = 0.8 - 0.04;
	const float Desaturation = 0.15;
	color *= toneMappingExposure;
	float x = min( color.r, min( color.g, color.b ) );
	float offset = x < 0.08 ? x - 6.25 * x * x : 0.04;
	color -= offset;
	float peak = max( color.r, max( color.g, color.b ) );
	if ( peak < StartCompression ) return color;
	float d = 1. - StartCompression;
	float newPeak = 1. - d * d / ( peak + d - StartCompression );
	color *= newPeak / peak;
	float g = 1. - 1. / ( Desaturation * ( peak - newPeak ) + 1. );
	return mix( color, vec3( newPeak ), g );
}
vec3 CustomToneMapping( vec3 color ) { return color; }`,Sg=`#ifdef USE_TRANSMISSION
	material.transmission = transmission;
	material.transmissionAlpha = 1.0;
	material.thickness = thickness;
	material.attenuationDistance = attenuationDistance;
	material.attenuationColor = attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		material.transmission *= texture2D( transmissionMap, vTransmissionMapUv ).r;
	#endif
	#ifdef USE_THICKNESSMAP
		material.thickness *= texture2D( thicknessMap, vThicknessMapUv ).g;
	#endif
	vec3 pos = vWorldPosition;
	vec3 v = normalize( cameraPosition - pos );
	vec3 n = transformNormalByInverseViewMatrix( normal, viewMatrix );
	vec4 transmitted = getIBLVolumeRefraction(
		n, v, material.roughness, material.diffuseContribution, material.specularColorBlended, material.specularF90,
		pos, modelMatrix, viewMatrix, projectionMatrix, material.dispersion, material.ior, material.thickness,
		material.attenuationColor, material.attenuationDistance );
	material.transmissionAlpha = mix( material.transmissionAlpha, transmitted.a, material.transmission );
	totalDiffuse = mix( totalDiffuse, transmitted.rgb, material.transmission );
#endif`,wg=`#ifdef USE_TRANSMISSION
	uniform float transmission;
	uniform float thickness;
	uniform float attenuationDistance;
	uniform vec3 attenuationColor;
	#ifdef USE_TRANSMISSIONMAP
		uniform sampler2D transmissionMap;
	#endif
	#ifdef USE_THICKNESSMAP
		uniform sampler2D thicknessMap;
	#endif
	uniform vec2 transmissionSamplerSize;
	uniform sampler2D transmissionSamplerMap;
	uniform mat4 modelMatrix;
	uniform mat4 projectionMatrix;
	varying vec3 vWorldPosition;
	float w0( float a ) {
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - a + 3.0 ) - 3.0 ) + 1.0 );
	}
	float w1( float a ) {
		return ( 1.0 / 6.0 ) * ( a *  a * ( 3.0 * a - 6.0 ) + 4.0 );
	}
	float w2( float a ){
		return ( 1.0 / 6.0 ) * ( a * ( a * ( - 3.0 * a + 3.0 ) + 3.0 ) + 1.0 );
	}
	float w3( float a ) {
		return ( 1.0 / 6.0 ) * ( a * a * a );
	}
	float g0( float a ) {
		return w0( a ) + w1( a );
	}
	float g1( float a ) {
		return w2( a ) + w3( a );
	}
	float h0( float a ) {
		return - 1.0 + w1( a ) / ( w0( a ) + w1( a ) );
	}
	float h1( float a ) {
		return 1.0 + w3( a ) / ( w2( a ) + w3( a ) );
	}
	vec4 bicubic( sampler2D tex, vec2 uv, vec4 texelSize, float lod ) {
		uv = uv * texelSize.zw + 0.5;
		vec2 iuv = floor( uv );
		vec2 fuv = fract( uv );
		float g0x = g0( fuv.x );
		float g1x = g1( fuv.x );
		float h0x = h0( fuv.x );
		float h1x = h1( fuv.x );
		float h0y = h0( fuv.y );
		float h1y = h1( fuv.y );
		vec2 p0 = ( vec2( iuv.x + h0x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p1 = ( vec2( iuv.x + h1x, iuv.y + h0y ) - 0.5 ) * texelSize.xy;
		vec2 p2 = ( vec2( iuv.x + h0x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		vec2 p3 = ( vec2( iuv.x + h1x, iuv.y + h1y ) - 0.5 ) * texelSize.xy;
		return g0( fuv.y ) * ( g0x * textureLod( tex, p0, lod ) + g1x * textureLod( tex, p1, lod ) ) +
			g1( fuv.y ) * ( g0x * textureLod( tex, p2, lod ) + g1x * textureLod( tex, p3, lod ) );
	}
	vec4 textureBicubic( sampler2D sampler, vec2 uv, float lod ) {
		vec2 fLodSize = vec2( textureSize( sampler, int( lod ) ) );
		vec2 cLodSize = vec2( textureSize( sampler, int( lod + 1.0 ) ) );
		vec2 fLodSizeInv = 1.0 / fLodSize;
		vec2 cLodSizeInv = 1.0 / cLodSize;
		vec4 fSample = bicubic( sampler, uv, vec4( fLodSizeInv, fLodSize ), floor( lod ) );
		vec4 cSample = bicubic( sampler, uv, vec4( cLodSizeInv, cLodSize ), ceil( lod ) );
		return mix( fSample, cSample, fract( lod ) );
	}
	vec3 getVolumeTransmissionRay( const in vec3 n, const in vec3 v, const in float thickness, const in float ior, const in mat4 modelMatrix ) {
		vec3 refractionVector = refract( - v, normalize( n ), 1.0 / ior );
		vec3 modelScale;
		modelScale.x = length( vec3( modelMatrix[ 0 ].xyz ) );
		modelScale.y = length( vec3( modelMatrix[ 1 ].xyz ) );
		modelScale.z = length( vec3( modelMatrix[ 2 ].xyz ) );
		return normalize( refractionVector ) * thickness * modelScale;
	}
	float applyIorToRoughness( const in float roughness, const in float ior ) {
		return roughness * clamp( ior * 2.0 - 2.0, 0.0, 1.0 );
	}
	vec4 getTransmissionSample( const in vec2 fragCoord, const in float roughness, const in float ior ) {
		float lod = log2( transmissionSamplerSize.x ) * applyIorToRoughness( roughness, ior );
		return textureBicubic( transmissionSamplerMap, fragCoord.xy, lod );
	}
	vec3 volumeAttenuation( const in float transmissionDistance, const in vec3 attenuationColor, const in float attenuationDistance ) {
		if ( isinf( attenuationDistance ) ) {
			return vec3( 1.0 );
		} else {
			vec3 attenuationCoefficient = -log( attenuationColor ) / attenuationDistance;
			vec3 transmittance = exp( - attenuationCoefficient * transmissionDistance );			return transmittance;
		}
	}
	vec4 getIBLVolumeRefraction( const in vec3 n, const in vec3 v, const in float roughness, const in vec3 diffuseColor,
		const in vec3 specularColor, const in float specularF90, const in vec3 position, const in mat4 modelMatrix,
		const in mat4 viewMatrix, const in mat4 projMatrix, const in float dispersion, const in float ior, const in float thickness,
		const in vec3 attenuationColor, const in float attenuationDistance ) {
		vec4 transmittedLight;
		vec3 transmittance;
		#ifdef USE_DISPERSION
			float halfSpread = ( ior - 1.0 ) * 0.025 * dispersion;
			vec3 iors = vec3( ior - halfSpread, ior, ior + halfSpread );
			for ( int i = 0; i < 3; i ++ ) {
				vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, iors[ i ], modelMatrix );
				vec3 refractedRayExit = position + transmissionRay;
				vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
				vec2 refractionCoords = ndcPos.xy / ndcPos.w;
				refractionCoords += 1.0;
				refractionCoords /= 2.0;
				vec4 transmissionSample = getTransmissionSample( refractionCoords, roughness, iors[ i ] );
				transmittedLight[ i ] = transmissionSample[ i ];
				transmittedLight.a += transmissionSample.a;
				transmittance[ i ] = diffuseColor[ i ] * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance )[ i ];
			}
			transmittedLight.a /= 3.0;
		#else
			vec3 transmissionRay = getVolumeTransmissionRay( n, v, thickness, ior, modelMatrix );
			vec3 refractedRayExit = position + transmissionRay;
			vec4 ndcPos = projMatrix * viewMatrix * vec4( refractedRayExit, 1.0 );
			vec2 refractionCoords = ndcPos.xy / ndcPos.w;
			refractionCoords += 1.0;
			refractionCoords /= 2.0;
			transmittedLight = getTransmissionSample( refractionCoords, roughness, ior );
			transmittance = diffuseColor * volumeAttenuation( length( transmissionRay ), attenuationColor, attenuationDistance );
		#endif
		vec3 attenuatedColor = transmittance * transmittedLight.rgb;
		vec3 F = EnvironmentBRDF( n, v, specularColor, specularF90, roughness );
		float transmittanceFactor = ( transmittance.r + transmittance.g + transmittance.b ) / 3.0;
		return vec4( ( 1.0 - F ) * attenuatedColor, 1.0 - ( 1.0 - transmittedLight.a ) * transmittanceFactor );
	}
#endif`,Tg=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_DIFFUSE_ROUGHNESSMAP
	varying vec2 vDiffuseRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_SPECULARMAP
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Ag=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	varying vec2 vUv;
#endif
#ifdef USE_MAP
	uniform mat3 mapTransform;
	varying vec2 vMapUv;
#endif
#ifdef USE_ALPHAMAP
	uniform mat3 alphaMapTransform;
	varying vec2 vAlphaMapUv;
#endif
#ifdef USE_LIGHTMAP
	uniform mat3 lightMapTransform;
	varying vec2 vLightMapUv;
#endif
#ifdef USE_AOMAP
	uniform mat3 aoMapTransform;
	varying vec2 vAoMapUv;
#endif
#ifdef USE_BUMPMAP
	uniform mat3 bumpMapTransform;
	varying vec2 vBumpMapUv;
#endif
#ifdef USE_NORMALMAP
	uniform mat3 normalMapTransform;
	varying vec2 vNormalMapUv;
#endif
#ifdef USE_DISPLACEMENTMAP
	uniform mat3 displacementMapTransform;
	varying vec2 vDisplacementMapUv;
#endif
#ifdef USE_EMISSIVEMAP
	uniform mat3 emissiveMapTransform;
	varying vec2 vEmissiveMapUv;
#endif
#ifdef USE_METALNESSMAP
	uniform mat3 metalnessMapTransform;
	varying vec2 vMetalnessMapUv;
#endif
#ifdef USE_ROUGHNESSMAP
	uniform mat3 roughnessMapTransform;
	varying vec2 vRoughnessMapUv;
#endif
#ifdef USE_DIFFUSE_ROUGHNESSMAP
	uniform mat3 diffuseRoughnessMapTransform;
	varying vec2 vDiffuseRoughnessMapUv;
#endif
#ifdef USE_ANISOTROPYMAP
	uniform mat3 anisotropyMapTransform;
	varying vec2 vAnisotropyMapUv;
#endif
#ifdef USE_CLEARCOATMAP
	uniform mat3 clearcoatMapTransform;
	varying vec2 vClearcoatMapUv;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	uniform mat3 clearcoatNormalMapTransform;
	varying vec2 vClearcoatNormalMapUv;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	uniform mat3 clearcoatRoughnessMapTransform;
	varying vec2 vClearcoatRoughnessMapUv;
#endif
#ifdef USE_SHEEN_COLORMAP
	uniform mat3 sheenColorMapTransform;
	varying vec2 vSheenColorMapUv;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	uniform mat3 sheenRoughnessMapTransform;
	varying vec2 vSheenRoughnessMapUv;
#endif
#ifdef USE_IRIDESCENCEMAP
	uniform mat3 iridescenceMapTransform;
	varying vec2 vIridescenceMapUv;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	uniform mat3 iridescenceThicknessMapTransform;
	varying vec2 vIridescenceThicknessMapUv;
#endif
#ifdef USE_SPECULARMAP
	uniform mat3 specularMapTransform;
	varying vec2 vSpecularMapUv;
#endif
#ifdef USE_SPECULAR_COLORMAP
	uniform mat3 specularColorMapTransform;
	varying vec2 vSpecularColorMapUv;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	uniform mat3 specularIntensityMapTransform;
	varying vec2 vSpecularIntensityMapUv;
#endif
#ifdef USE_TRANSMISSIONMAP
	uniform mat3 transmissionMapTransform;
	varying vec2 vTransmissionMapUv;
#endif
#ifdef USE_THICKNESSMAP
	uniform mat3 thicknessMapTransform;
	varying vec2 vThicknessMapUv;
#endif`,Eg=`#if defined( USE_UV ) || defined( USE_ANISOTROPY )
	vUv = vec3( uv, 1 ).xy;
#endif
#ifdef USE_MAP
	vMapUv = ( mapTransform * vec3( MAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ALPHAMAP
	vAlphaMapUv = ( alphaMapTransform * vec3( ALPHAMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_LIGHTMAP
	vLightMapUv = ( lightMapTransform * vec3( LIGHTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_AOMAP
	vAoMapUv = ( aoMapTransform * vec3( AOMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_BUMPMAP
	vBumpMapUv = ( bumpMapTransform * vec3( BUMPMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_NORMALMAP
	vNormalMapUv = ( normalMapTransform * vec3( NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DISPLACEMENTMAP
	vDisplacementMapUv = ( displacementMapTransform * vec3( DISPLACEMENTMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_EMISSIVEMAP
	vEmissiveMapUv = ( emissiveMapTransform * vec3( EMISSIVEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_METALNESSMAP
	vMetalnessMapUv = ( metalnessMapTransform * vec3( METALNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ROUGHNESSMAP
	vRoughnessMapUv = ( roughnessMapTransform * vec3( ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_DIFFUSE_ROUGHNESSMAP
	vDiffuseRoughnessMapUv = ( diffuseRoughnessMapTransform * vec3( DIFFUSE_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_ANISOTROPYMAP
	vAnisotropyMapUv = ( anisotropyMapTransform * vec3( ANISOTROPYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOATMAP
	vClearcoatMapUv = ( clearcoatMapTransform * vec3( CLEARCOATMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_NORMALMAP
	vClearcoatNormalMapUv = ( clearcoatNormalMapTransform * vec3( CLEARCOAT_NORMALMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_CLEARCOAT_ROUGHNESSMAP
	vClearcoatRoughnessMapUv = ( clearcoatRoughnessMapTransform * vec3( CLEARCOAT_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCEMAP
	vIridescenceMapUv = ( iridescenceMapTransform * vec3( IRIDESCENCEMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_IRIDESCENCE_THICKNESSMAP
	vIridescenceThicknessMapUv = ( iridescenceThicknessMapTransform * vec3( IRIDESCENCE_THICKNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_COLORMAP
	vSheenColorMapUv = ( sheenColorMapTransform * vec3( SHEEN_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SHEEN_ROUGHNESSMAP
	vSheenRoughnessMapUv = ( sheenRoughnessMapTransform * vec3( SHEEN_ROUGHNESSMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULARMAP
	vSpecularMapUv = ( specularMapTransform * vec3( SPECULARMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_COLORMAP
	vSpecularColorMapUv = ( specularColorMapTransform * vec3( SPECULAR_COLORMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_SPECULAR_INTENSITYMAP
	vSpecularIntensityMapUv = ( specularIntensityMapTransform * vec3( SPECULAR_INTENSITYMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_TRANSMISSIONMAP
	vTransmissionMapUv = ( transmissionMapTransform * vec3( TRANSMISSIONMAP_UV, 1 ) ).xy;
#endif
#ifdef USE_THICKNESSMAP
	vThicknessMapUv = ( thicknessMapTransform * vec3( THICKNESSMAP_UV, 1 ) ).xy;
#endif`,Cg=`#if defined( USE_ENVMAP ) || defined( DISTANCE ) || defined ( USE_SHADOWMAP ) || defined ( USE_TRANSMISSION ) || NUM_SPOT_LIGHT_COORDS > 0
	vec4 worldPosition = vec4( transformed, 1.0 );
	#ifdef USE_BATCHING
		worldPosition = batchingMatrix * worldPosition;
	#endif
	#ifdef USE_INSTANCING
		worldPosition = instanceMatrix * worldPosition;
	#endif
	worldPosition = modelMatrix * worldPosition;
#endif`,Rg=`varying vec2 vUv;
uniform mat3 uvTransform;
void main() {
	vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	gl_Position = vec4( position.xy, 1.0, 1.0 );
}`,Pg=`uniform sampler2D t2D;
uniform float backgroundIntensity;
varying vec2 vUv;
void main() {
	vec4 texColor = texture2D( t2D, vUv );
	#ifdef DECODE_VIDEO_TEXTURE
		texColor = vec4( mix( pow( texColor.rgb * 0.9478672986 + vec3( 0.0521327014 ), vec3( 2.4 ) ), texColor.rgb * 0.0773993808, vec3( lessThanEqual( texColor.rgb, vec3( 0.04045 ) ) ) ), texColor.w );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Ig=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Lg=`uniform float backgroundBlurriness;
uniform float backgroundIntensity;
uniform mat3 backgroundRotation;
varying vec3 vWorldDirection;
#include <envmap_common_pars_fragment>
void main() {
	#ifdef ENVMAP_TYPE_PMREM
		vec4 texColor = textureLod( envMap, backgroundRotation * vWorldDirection, roughnessToMip( backgroundBlurriness ) );
	#else
		vec4 texColor = textureCube( envMap, backgroundRotation * vWorldDirection );
	#endif
	texColor.rgb *= backgroundIntensity;
	gl_FragColor = texColor;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Dg=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
	gl_Position.z = gl_Position.w;
}`,Ng=`uniform samplerCube tCube;
uniform float tFlip;
uniform float opacity;
varying vec3 vWorldDirection;
void main() {
	vec4 texColor = textureCube( tCube, vec3( tFlip * vWorldDirection.x, vWorldDirection.yz ) );
	gl_FragColor = texColor;
	gl_FragColor.a *= opacity;
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Ug=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
varying vec2 vHighPrecisionZW;
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vHighPrecisionZW = gl_Position.zw;
}`,Fg=`#if DEPTH_PACKING == 3200
	uniform float opacity;
#endif
#include <common>
#include <packing>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
varying vec2 vHighPrecisionZW;
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#if DEPTH_PACKING == 3200
		diffuseColor.a = opacity;
	#endif
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <logdepthbuf_fragment>
	#ifdef USE_REVERSED_DEPTH_BUFFER
		float fragCoordZ = vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ];
	#else
		float fragCoordZ = 0.5 * vHighPrecisionZW[ 0 ] / vHighPrecisionZW[ 1 ] + 0.5;
	#endif
	#if DEPTH_PACKING == 3200
		gl_FragColor = vec4( vec3( 1.0 - fragCoordZ ), opacity );
	#elif DEPTH_PACKING == 3201
		gl_FragColor = packDepthToRGBA( fragCoordZ );
	#elif DEPTH_PACKING == 3202
		gl_FragColor = vec4( packDepthToRGB( fragCoordZ ), 1.0 );
	#elif DEPTH_PACKING == 3203
		gl_FragColor = vec4( packDepthToRG( fragCoordZ ), 0.0, 1.0 );
	#endif
}`,Og=`#define DISTANCE
varying vec3 vWorldPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <skinbase_vertex>
	#include <morphinstance_vertex>
	#ifdef USE_DISPLACEMENTMAP
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <worldpos_vertex>
	#include <clipping_planes_vertex>
	vWorldPosition = worldPosition.xyz;
}`,Bg=`#define DISTANCE
uniform vec3 referencePosition;
uniform float nearDistance;
uniform float farDistance;
varying vec3 vWorldPosition;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 1.0 );
	#include <clipping_planes_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	float dist = length( vWorldPosition - referencePosition );
	dist = ( dist - nearDistance ) / ( farDistance - nearDistance );
	dist = saturate( dist );
	gl_FragColor = vec4( dist, 0.0, 0.0, 1.0 );
}`,kg=`varying vec3 vWorldDirection;
#include <common>
void main() {
	vWorldDirection = transformDirection( position, modelMatrix );
	#include <begin_vertex>
	#include <project_vertex>
}`,zg=`uniform sampler2D tEquirect;
varying vec3 vWorldDirection;
#include <common>
void main() {
	vec3 direction = normalize( vWorldDirection );
	vec2 sampleUV = equirectUv( direction );
	gl_FragColor = texture2D( tEquirect, sampleUV );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
}`,Gg=`uniform float scale;
attribute float lineDistance;
varying float vLineDistance;
#include <common>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	vLineDistance = scale * lineDistance;
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,Vg=`uniform vec3 diffuse;
uniform float opacity;
uniform float dashSize;
uniform float totalSize;
varying float vLineDistance;
#include <common>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	if ( mod( vLineDistance, totalSize ) > dashSize ) {
		discard;
	}
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,Hg=`#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#if defined ( USE_ENVMAP ) || defined ( USE_SKINNING )
		#include <beginnormal_vertex>
		#include <morphnormal_vertex>
		#include <skinbase_vertex>
		#include <skinnormal_vertex>
		#include <defaultnormal_vertex>
	#endif
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <fog_vertex>
}`,Wg=`uniform vec3 diffuse;
uniform float opacity;
#ifndef FLAT_SHADED
	varying vec3 vNormal;
#endif
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <fog_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	#ifdef USE_LIGHTMAP
		vec4 lightMapTexel = texture2D( lightMap, vLightMapUv );
		reflectedLight.indirectDiffuse += lightMapTexel.rgb * lightMapIntensity * RECIPROCAL_PI;
	#else
		reflectedLight.indirectDiffuse += vec3( 1.0 );
	#endif
	#include <aomap_fragment>
	reflectedLight.indirectDiffuse *= diffuseColor.rgb;
	vec3 outgoingLight = reflectedLight.indirectDiffuse;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Xg=`#define LAMBERT
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,qg=`#define LAMBERT
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_lambert_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_lambert_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Yg=`#define MATCAP
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <color_pars_vertex>
#include <displacementmap_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
	vViewPosition = - mvPosition.xyz;
}`,$g=`#define MATCAP
uniform vec3 diffuse;
uniform float opacity;
uniform sampler2D matcap;
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	vec3 viewDir = normalize( vViewPosition );
	vec3 x = normalize( vec3( viewDir.z, 0.0, - viewDir.x ) );
	vec3 y = cross( viewDir, x );
	vec2 uv = vec2( dot( x, normal ), dot( y, normal ) ) * 0.495 + 0.5;
	#ifdef USE_MATCAP
		vec4 matcapColor = texture2D( matcap, uv );
	#else
		vec4 matcapColor = vec4( vec3( mix( 0.2, 0.8, uv.y ) ), 1.0 );
	#endif
	vec3 outgoingLight = diffuseColor.rgb * matcapColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Zg=`#define NORMAL
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	vViewPosition = - mvPosition.xyz;
#endif
}`,Jg=`#define NORMAL
uniform float opacity;
#if defined( FLAT_SHADED ) || defined( USE_BUMPMAP ) || defined( USE_NORMALMAP_TANGENTSPACE )
	varying vec3 vViewPosition;
#endif
#include <uv_pars_fragment>
#include <normal_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( 0.0, 0.0, 0.0, opacity );
	#include <clipping_planes_fragment>
	#include <logdepthbuf_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	gl_FragColor = vec4( normalize( normal ) * 0.5 + 0.5, diffuseColor.a );
	#ifdef OPAQUE
		gl_FragColor.a = 1.0;
	#endif
}`,Kg=`#define PHONG
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <envmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <envmap_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,jg=`#define PHONG
uniform vec3 diffuse;
uniform vec3 emissive;
uniform vec3 specular;
uniform float shininess;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_phong_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <specularmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <specularmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_phong_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + reflectedLight.directSpecular + reflectedLight.indirectSpecular + totalEmissiveRadiance;
	#include <envmap_fragment>
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,Qg=`#define STANDARD
varying vec3 vViewPosition;
#ifdef USE_TRANSMISSION
	varying vec3 vWorldPosition;
#endif
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
#ifdef USE_TRANSMISSION
	vWorldPosition = worldPosition.xyz;
#endif
}`,t0=`#define STANDARD
#ifdef PHYSICAL
	#define IOR
	#define USE_SPECULAR
#endif
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float roughness;
uniform float metalness;
uniform float opacity;
#ifdef IOR
	uniform float ior;
#endif
#ifdef USE_SPECULAR
	uniform float specularIntensity;
	uniform vec3 specularColor;
	#ifdef USE_SPECULAR_COLORMAP
		uniform sampler2D specularColorMap;
	#endif
	#ifdef USE_SPECULAR_INTENSITYMAP
		uniform sampler2D specularIntensityMap;
	#endif
#endif
#ifdef USE_CLEARCOAT
	uniform float clearcoat;
	uniform float clearcoatRoughness;
#endif
#ifdef USE_DIFFUSE_ROUGHNESS
	uniform float diffuseRoughness;
	#ifdef USE_DIFFUSE_ROUGHNESSMAP
		uniform sampler2D diffuseRoughnessMap;
	#endif
#endif
#ifdef USE_DISPERSION
	uniform float dispersion;
#endif
#ifdef USE_RETROREFLECTION
	uniform float retroreflectivity;
#endif
#ifdef USE_IRIDESCENCE
	uniform float iridescence;
	uniform float iridescenceIOR;
	uniform float iridescenceThicknessMinimum;
	uniform float iridescenceThicknessMaximum;
#endif
#ifdef USE_SHEEN
	uniform vec3 sheenColor;
	uniform float sheenRoughness;
	#ifdef USE_SHEEN_COLORMAP
		uniform sampler2D sheenColorMap;
	#endif
	#ifdef USE_SHEEN_ROUGHNESSMAP
		uniform sampler2D sheenRoughnessMap;
	#endif
#endif
#ifdef USE_ANISOTROPY
	uniform vec2 anisotropyVector;
	#ifdef USE_ANISOTROPYMAP
		uniform sampler2D anisotropyMap;
	#endif
#endif
varying vec3 vViewPosition;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <iridescence_fragment>
#include <envmap_common_pars_fragment>
#include <envmap_physical_pars_fragment>
#include <fog_pars_fragment>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_physical_pars_fragment>
#include <transmission_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <clearcoat_pars_fragment>
#include <iridescence_pars_fragment>
#include <roughnessmap_pars_fragment>
#include <metalnessmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <roughnessmap_fragment>
	#include <metalnessmap_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <clearcoat_normal_fragment_begin>
	#include <clearcoat_normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_physical_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 totalDiffuse = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse;
	vec3 totalSpecular = reflectedLight.directSpecular + reflectedLight.indirectSpecular;
	#include <transmission_fragment>
	vec3 outgoingLight = totalDiffuse + totalSpecular + totalEmissiveRadiance;
	#ifdef USE_SHEEN
 
		outgoingLight = outgoingLight + sheenSpecularDirect + sheenSpecularIndirect;
 
 	#endif
	#ifdef USE_CLEARCOAT
		float dotNVcc = saturate( dot( geometryClearcoatNormal, geometryViewDir ) );
		vec3 Fcc = F_Schlick( material.clearcoatF0, material.clearcoatF90, dotNVcc );
		outgoingLight = outgoingLight * ( 1.0 - material.clearcoat * Fcc ) + ( clearcoatSpecularDirect + clearcoatSpecularIndirect ) * material.clearcoat;
	#endif
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,e0=`#define TOON
varying vec3 vViewPosition;
#include <common>
#include <batching_pars_vertex>
#include <uv_pars_vertex>
#include <displacementmap_pars_vertex>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <normal_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <shadowmap_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <normal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <displacementmap_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	vViewPosition = - mvPosition.xyz;
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,n0=`#define TOON
uniform vec3 diffuse;
uniform vec3 emissive;
uniform float opacity;
#include <common>
#include <dithering_pars_fragment>
#include <color_pars_fragment>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <aomap_pars_fragment>
#include <lightmap_pars_fragment>
#include <emissivemap_pars_fragment>
#include <gradientmap_pars_fragment>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <normal_pars_fragment>
#include <lights_toon_pars_fragment>
#include <shadowmap_pars_fragment>
#include <bumpmap_pars_fragment>
#include <normalmap_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	ReflectedLight reflectedLight = ReflectedLight( vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ), vec3( 0.0 ) );
	vec3 totalEmissiveRadiance = emissive;
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <color_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	#include <normal_fragment_begin>
	#include <normal_fragment_maps>
	#include <emissivemap_fragment>
	#include <lights_toon_fragment>
	#include <lights_fragment_begin>
	#include <lights_fragment_maps>
	#include <lights_fragment_end>
	#include <aomap_fragment>
	vec3 outgoingLight = reflectedLight.directDiffuse + reflectedLight.indirectDiffuse + totalEmissiveRadiance;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
	#include <dithering_fragment>
}`,i0=`uniform float size;
uniform float scale;
#include <common>
#include <color_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
#ifdef USE_POINTS_UV
	varying vec2 vUv;
	uniform mat3 uvTransform;
#endif
void main() {
	#ifdef USE_POINTS_UV
		vUv = ( uvTransform * vec3( uv, 1 ) ).xy;
	#endif
	#include <color_vertex>
	#include <morphinstance_vertex>
	#include <morphcolor_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <project_vertex>
	gl_PointSize = size;
	#ifdef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) gl_PointSize *= ( scale / - mvPosition.z );
	#endif
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <worldpos_vertex>
	#include <fog_vertex>
}`,s0=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <color_pars_fragment>
#include <map_particle_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_particle_fragment>
	#include <color_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,r0=`#include <common>
#include <batching_pars_vertex>
#include <fog_pars_vertex>
#include <morphtarget_pars_vertex>
#include <skinning_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <shadowmap_pars_vertex>
void main() {
	#include <batching_vertex>
	#include <beginnormal_vertex>
	#include <morphinstance_vertex>
	#include <morphnormal_vertex>
	#include <skinbase_vertex>
	#include <skinnormal_vertex>
	#include <defaultnormal_vertex>
	#include <begin_vertex>
	#include <morphtarget_vertex>
	#include <skinning_vertex>
	#include <project_vertex>
	#include <logdepthbuf_vertex>
	#include <worldpos_vertex>
	#include <shadowmap_vertex>
	#include <fog_vertex>
}`,a0=`uniform vec3 color;
uniform float opacity;
#include <common>
#include <fog_pars_fragment>
#include <bsdfs>
#include <lights_pars_begin>
#include <logdepthbuf_pars_fragment>
#include <shadowmap_pars_fragment>
#include <shadowmask_pars_fragment>
void main() {
	#include <logdepthbuf_fragment>
	gl_FragColor = vec4( color, opacity * ( 1.0 - getShadowMask() ) );
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
	#include <premultiplied_alpha_fragment>
}`,o0=`uniform float rotation;
uniform vec2 center;
#include <common>
#include <uv_pars_vertex>
#include <fog_pars_vertex>
#include <logdepthbuf_pars_vertex>
#include <clipping_planes_pars_vertex>
void main() {
	#include <uv_vertex>
	vec4 mvPosition = modelViewMatrix[ 3 ];
	vec2 scale = vec2( length( modelMatrix[ 0 ].xyz ), length( modelMatrix[ 1 ].xyz ) );
	#ifndef USE_SIZEATTENUATION
		bool isPerspective = isPerspectiveMatrix( projectionMatrix );
		if ( isPerspective ) scale *= - mvPosition.z;
	#endif
	vec2 alignedPosition = ( position.xy - ( center - vec2( 0.5 ) ) ) * scale;
	vec2 rotatedPosition;
	rotatedPosition.x = cos( rotation ) * alignedPosition.x - sin( rotation ) * alignedPosition.y;
	rotatedPosition.y = sin( rotation ) * alignedPosition.x + cos( rotation ) * alignedPosition.y;
	mvPosition.xy += rotatedPosition;
	gl_Position = projectionMatrix * mvPosition;
	#include <logdepthbuf_vertex>
	#include <clipping_planes_vertex>
	#include <fog_vertex>
}`,l0=`uniform vec3 diffuse;
uniform float opacity;
#include <common>
#include <uv_pars_fragment>
#include <map_pars_fragment>
#include <alphamap_pars_fragment>
#include <alphatest_pars_fragment>
#include <alphahash_pars_fragment>
#include <fog_pars_fragment>
#include <logdepthbuf_pars_fragment>
#include <clipping_planes_pars_fragment>
void main() {
	vec4 diffuseColor = vec4( diffuse, opacity );
	#include <clipping_planes_fragment>
	vec3 outgoingLight = vec3( 0.0 );
	#include <logdepthbuf_fragment>
	#include <map_fragment>
	#include <alphamap_fragment>
	#include <alphatest_fragment>
	#include <alphahash_fragment>
	outgoingLight = diffuseColor.rgb;
	#include <opaque_fragment>
	#include <tonemapping_fragment>
	#include <colorspace_fragment>
	#include <fog_fragment>
}`,ie={alphahash_fragment:Pp,alphahash_pars_fragment:Ip,alphamap_fragment:Lp,alphamap_pars_fragment:Dp,alphatest_fragment:Np,alphatest_pars_fragment:Up,aomap_fragment:Fp,aomap_pars_fragment:Op,batching_pars_vertex:Bp,batching_vertex:kp,begin_vertex:zp,beginnormal_vertex:Gp,bsdfs:Vp,iridescence_fragment:Hp,bumpmap_pars_fragment:Wp,clipping_planes_fragment:Xp,clipping_planes_pars_fragment:qp,clipping_planes_pars_vertex:Yp,clipping_planes_vertex:$p,color_fragment:Zp,color_pars_fragment:Jp,color_pars_vertex:Kp,color_vertex:jp,common:Qp,defaultnormal_vertex:tm,displacementmap_pars_vertex:em,displacementmap_vertex:nm,emissivemap_fragment:im,emissivemap_pars_fragment:sm,colorspace_fragment:rm,colorspace_pars_fragment:am,envmap_fragment:om,envmap_common_pars_fragment:lm,envmap_pars_fragment:cm,envmap_pars_vertex:hm,envmap_physical_pars_fragment:bm,envmap_vertex:um,fog_vertex:dm,fog_pars_vertex:fm,fog_fragment:pm,fog_pars_fragment:mm,gradientmap_pars_fragment:gm,lightmap_pars_fragment:xm,lights_lambert_fragment:_m,lights_lambert_pars_fragment:ym,lights_pars_begin:vm,lights_toon_fragment:Mm,lights_toon_pars_fragment:Sm,lights_phong_fragment:wm,lights_phong_pars_fragment:Tm,lights_physical_fragment:Am,lights_physical_pars_fragment:Em,lights_fragment_begin:Cm,lights_fragment_maps:Rm,lights_fragment_end:Pm,lightprobes_pars_fragment:Im,logdepthbuf_fragment:Lm,logdepthbuf_pars_fragment:Dm,logdepthbuf_pars_vertex:Nm,logdepthbuf_vertex:Um,map_fragment:Fm,map_pars_fragment:Om,map_particle_fragment:Bm,map_particle_pars_fragment:km,metalnessmap_fragment:zm,metalnessmap_pars_fragment:Gm,morphinstance_vertex:Vm,morphcolor_vertex:Hm,morphnormal_vertex:Wm,morphtarget_pars_vertex:Xm,morphtarget_vertex:qm,normal_fragment_begin:Ym,normal_fragment_maps:$m,normal_pars_fragment:Zm,normal_pars_vertex:Jm,normal_vertex:Km,normalmap_pars_fragment:jm,clearcoat_normal_fragment_begin:Qm,clearcoat_normal_fragment_maps:tg,clearcoat_pars_fragment:eg,iridescence_pars_fragment:ng,opaque_fragment:ig,packing:sg,premultiplied_alpha_fragment:rg,project_vertex:ag,dithering_fragment:og,dithering_pars_fragment:lg,roughnessmap_fragment:cg,roughnessmap_pars_fragment:hg,shadowmap_pars_fragment:ug,shadowmap_pars_vertex:dg,shadowmap_vertex:fg,shadowmask_pars_fragment:pg,skinbase_vertex:mg,skinning_pars_vertex:gg,skinning_vertex:xg,skinnormal_vertex:_g,specularmap_fragment:yg,specularmap_pars_fragment:vg,tonemapping_fragment:bg,tonemapping_pars_fragment:Mg,transmission_fragment:Sg,transmission_pars_fragment:wg,uv_pars_fragment:Tg,uv_pars_vertex:Ag,uv_vertex:Eg,worldpos_vertex:Cg,background_vert:Rg,background_frag:Pg,backgroundCube_vert:Ig,backgroundCube_frag:Lg,cube_vert:Dg,cube_frag:Ng,depth_vert:Ug,depth_frag:Fg,distance_vert:Og,distance_frag:Bg,equirect_vert:kg,equirect_frag:zg,linedashed_vert:Gg,linedashed_frag:Vg,meshbasic_vert:Hg,meshbasic_frag:Wg,meshlambert_vert:Xg,meshlambert_frag:qg,meshmatcap_vert:Yg,meshmatcap_frag:$g,meshnormal_vert:Zg,meshnormal_frag:Jg,meshphong_vert:Kg,meshphong_frag:jg,meshphysical_vert:Qg,meshphysical_frag:t0,meshtoon_vert:e0,meshtoon_frag:n0,points_vert:i0,points_frag:s0,shadow_vert:r0,shadow_frag:a0,sprite_vert:o0,sprite_frag:l0},Ut={common:{diffuse:{value:new Vt(16777215)},opacity:{value:1},map:{value:null},mapTransform:{value:new Xt},alphaMap:{value:null},alphaMapTransform:{value:new Xt},alphaTest:{value:0}},specularmap:{specularMap:{value:null},specularMapTransform:{value:new Xt}},envmap:{envMap:{value:null},envMapRotation:{value:new Xt},reflectivity:{value:1},ior:{value:1.5},refractionRatio:{value:.98},dfgLUT:{value:null}},aomap:{aoMap:{value:null},aoMapIntensity:{value:1},aoMapTransform:{value:new Xt}},lightmap:{lightMap:{value:null},lightMapIntensity:{value:1},lightMapTransform:{value:new Xt}},bumpmap:{bumpMap:{value:null},bumpMapTransform:{value:new Xt},bumpScale:{value:1}},normalmap:{normalMap:{value:null},normalMapTransform:{value:new Xt},normalScale:{value:new ft(1,1)}},displacementmap:{displacementMap:{value:null},displacementMapTransform:{value:new Xt},displacementScale:{value:1},displacementBias:{value:0}},emissivemap:{emissiveMap:{value:null},emissiveMapTransform:{value:new Xt}},metalnessmap:{metalnessMap:{value:null},metalnessMapTransform:{value:new Xt}},roughnessmap:{roughnessMap:{value:null},roughnessMapTransform:{value:new Xt}},gradientmap:{gradientMap:{value:null}},fog:{fogDensity:{value:25e-5},fogNear:{value:1},fogFar:{value:2e3},fogColor:{value:new Vt(16777215)}},lights:{ambientLightColor:{value:[]},lightProbe:{value:[]},sunLights:{value:[],properties:{direction:{},color:{}}},sunLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},sunShadowMatrix:{value:[]},sunShadowCascade:{value:[]},directionalLights:{value:[],properties:{direction:{},color:{}}},directionalLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},directionalShadowMatrix:{value:[]},spotLights:{value:[],properties:{color:{},position:{},direction:{},distance:{},coneCos:{},penumbraCos:{},decay:{}}},spotLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{}}},spotLightMap:{value:[]},spotLightMatrix:{value:[]},pointLights:{value:[],properties:{color:{},position:{},decay:{},distance:{}}},pointLightShadows:{value:[],properties:{shadowIntensity:1,shadowBias:{},shadowNormalBias:{},shadowRadius:{},shadowMapSize:{},shadowCameraNear:{},shadowCameraFar:{}}},pointShadowMatrix:{value:[]},hemisphereLights:{value:[],properties:{direction:{},skyColor:{},groundColor:{}}},rectAreaLights:{value:[],properties:{color:{},position:{},width:{},height:{}}},ltc_1:{value:null},ltc_2:{value:null},probesSH:{value:[]},probesMin:{value:[]},probesMax:{value:[]},probesResolution:{value:[]}},points:{diffuse:{value:new Vt(16777215)},opacity:{value:1},size:{value:1},scale:{value:1},map:{value:null},alphaMap:{value:null},alphaMapTransform:{value:new Xt},alphaTest:{value:0},uvTransform:{value:new Xt}},sprite:{diffuse:{value:new Vt(16777215)},opacity:{value:1},center:{value:new ft(.5,.5)},rotation:{value:0},map:{value:null},mapTransform:{value:new Xt},alphaMap:{value:null},alphaMapTransform:{value:new Xt},alphaTest:{value:0}}},zn={basic:{uniforms:Ze([Ut.common,Ut.specularmap,Ut.envmap,Ut.aomap,Ut.lightmap,Ut.fog]),vertexShader:ie.meshbasic_vert,fragmentShader:ie.meshbasic_frag},lambert:{uniforms:Ze([Ut.common,Ut.specularmap,Ut.envmap,Ut.aomap,Ut.lightmap,Ut.emissivemap,Ut.bumpmap,Ut.normalmap,Ut.displacementmap,Ut.fog,Ut.lights,{emissive:{value:new Vt(0)},envMapIntensity:{value:1}}]),vertexShader:ie.meshlambert_vert,fragmentShader:ie.meshlambert_frag},phong:{uniforms:Ze([Ut.common,Ut.specularmap,Ut.envmap,Ut.aomap,Ut.lightmap,Ut.emissivemap,Ut.bumpmap,Ut.normalmap,Ut.displacementmap,Ut.fog,Ut.lights,{emissive:{value:new Vt(0)},specular:{value:new Vt(1118481)},shininess:{value:30},envMapIntensity:{value:1}}]),vertexShader:ie.meshphong_vert,fragmentShader:ie.meshphong_frag},standard:{uniforms:Ze([Ut.common,Ut.envmap,Ut.aomap,Ut.lightmap,Ut.emissivemap,Ut.bumpmap,Ut.normalmap,Ut.displacementmap,Ut.roughnessmap,Ut.metalnessmap,Ut.fog,Ut.lights,{emissive:{value:new Vt(0)},roughness:{value:1},metalness:{value:0},envMapIntensity:{value:1}}]),vertexShader:ie.meshphysical_vert,fragmentShader:ie.meshphysical_frag},toon:{uniforms:Ze([Ut.common,Ut.aomap,Ut.lightmap,Ut.emissivemap,Ut.bumpmap,Ut.normalmap,Ut.displacementmap,Ut.gradientmap,Ut.fog,Ut.lights,{emissive:{value:new Vt(0)}}]),vertexShader:ie.meshtoon_vert,fragmentShader:ie.meshtoon_frag},matcap:{uniforms:Ze([Ut.common,Ut.bumpmap,Ut.normalmap,Ut.displacementmap,Ut.fog,{matcap:{value:null}}]),vertexShader:ie.meshmatcap_vert,fragmentShader:ie.meshmatcap_frag},points:{uniforms:Ze([Ut.points,Ut.fog]),vertexShader:ie.points_vert,fragmentShader:ie.points_frag},dashed:{uniforms:Ze([Ut.common,Ut.fog,{scale:{value:1},dashSize:{value:1},totalSize:{value:2}}]),vertexShader:ie.linedashed_vert,fragmentShader:ie.linedashed_frag},depth:{uniforms:Ze([Ut.common,Ut.displacementmap]),vertexShader:ie.depth_vert,fragmentShader:ie.depth_frag},normal:{uniforms:Ze([Ut.common,Ut.bumpmap,Ut.normalmap,Ut.displacementmap,{opacity:{value:1}}]),vertexShader:ie.meshnormal_vert,fragmentShader:ie.meshnormal_frag},sprite:{uniforms:Ze([Ut.sprite,Ut.fog]),vertexShader:ie.sprite_vert,fragmentShader:ie.sprite_frag},background:{uniforms:{uvTransform:{value:new Xt},t2D:{value:null},backgroundIntensity:{value:1}},vertexShader:ie.background_vert,fragmentShader:ie.background_frag},backgroundCube:{uniforms:{envMap:{value:null},backgroundBlurriness:{value:0},backgroundIntensity:{value:1},backgroundRotation:{value:new Xt}},vertexShader:ie.backgroundCube_vert,fragmentShader:ie.backgroundCube_frag},cube:{uniforms:{tCube:{value:null},tFlip:{value:-1},opacity:{value:1}},vertexShader:ie.cube_vert,fragmentShader:ie.cube_frag},equirect:{uniforms:{tEquirect:{value:null}},vertexShader:ie.equirect_vert,fragmentShader:ie.equirect_frag},distance:{uniforms:Ze([Ut.common,Ut.displacementmap,{referencePosition:{value:new F},nearDistance:{value:1},farDistance:{value:1e3}}]),vertexShader:ie.distance_vert,fragmentShader:ie.distance_frag},shadow:{uniforms:Ze([Ut.lights,Ut.fog,{color:{value:new Vt(0)},opacity:{value:1}}]),vertexShader:ie.shadow_vert,fragmentShader:ie.shadow_frag}};zn.physical={uniforms:Ze([zn.standard.uniforms,{clearcoat:{value:0},clearcoatMap:{value:null},clearcoatMapTransform:{value:new Xt},clearcoatNormalMap:{value:null},clearcoatNormalMapTransform:{value:new Xt},clearcoatNormalScale:{value:new ft(1,1)},clearcoatRoughness:{value:0},clearcoatRoughnessMap:{value:null},clearcoatRoughnessMapTransform:{value:new Xt},diffuseRoughness:{value:0},diffuseRoughnessMap:{value:null},diffuseRoughnessMapTransform:{value:new Xt},dispersion:{value:0},retroreflectivity:{value:0},iridescence:{value:0},iridescenceMap:{value:null},iridescenceMapTransform:{value:new Xt},iridescenceIOR:{value:1.3},iridescenceThicknessMinimum:{value:100},iridescenceThicknessMaximum:{value:400},iridescenceThicknessMap:{value:null},iridescenceThicknessMapTransform:{value:new Xt},sheen:{value:0},sheenColor:{value:new Vt(0)},sheenColorMap:{value:null},sheenColorMapTransform:{value:new Xt},sheenRoughness:{value:1},sheenRoughnessMap:{value:null},sheenRoughnessMapTransform:{value:new Xt},transmission:{value:0},transmissionMap:{value:null},transmissionMapTransform:{value:new Xt},transmissionSamplerSize:{value:new ft},transmissionSamplerMap:{value:null},thickness:{value:0},thicknessMap:{value:null},thicknessMapTransform:{value:new Xt},attenuationDistance:{value:0},attenuationColor:{value:new Vt(0)},specularColor:{value:new Vt(1,1,1)},specularColorMap:{value:null},specularColorMapTransform:{value:new Xt},specularIntensity:{value:1},specularIntensityMap:{value:null},specularIntensityMapTransform:{value:new Xt},anisotropyVector:{value:new ft},anisotropyMap:{value:null},anisotropyMapTransform:{value:new Xt}}]),vertexShader:ie.meshphysical_vert,fragmentShader:ie.meshphysical_frag};var $o={r:0,b:0,g:0},c0=new xe,cd=new Xt;cd.set(-1,0,0,0,1,0,0,0,1);function h0(i,t,e,n,s,r){let a=new Vt(0),o=s===!0?0:1,c,l,u=null,h=0,d=null;function f(v){let b=v.isScene===!0?v.background:null;if(b&&b.isTexture){let _=v.backgroundBlurriness>0;b=t.get(b,_)}return b}function m(v){let b=!1,_=f(v);_===null?g(a,o):_&&_.isColor&&(g(_,1),b=!0);let T=i.xr.getEnvironmentBlendMode();T==="additive"?e.buffers.color.setClear(0,0,0,1,r):T==="alpha-blend"&&e.buffers.color.setClear(0,0,0,0,r),(i.autoClear||b)&&(e.buffers.depth.setTest(!0),e.buffers.depth.setMask(!0),e.buffers.color.setMask(!0),i.clear(i.autoClearColor,i.autoClearDepth,i.autoClearStencil))}function y(v,b){let _=f(b);_&&_.isCubeTexture?(l===void 0&&(l=new he(new Ce(1,1,1),new dn({name:"BackgroundCubeMaterial",uniforms:Ji(zn.backgroundCube.uniforms),vertexShader:zn.backgroundCube.vertexShader,fragmentShader:zn.backgroundCube.fragmentShader,side:We,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),l.geometry.deleteAttribute("normal"),l.geometry.deleteAttribute("uv"),l.onBeforeRender=function(T,w,x){this.matrixWorld.copyPosition(x.matrixWorld)},Object.defineProperty(l.material,"envMap",{get:function(){return this.uniforms.envMap.value}}),n.update(l)),l.material.uniforms.envMap.value=_,l.material.uniforms.backgroundBlurriness.value=b.backgroundBlurriness,l.material.uniforms.backgroundIntensity.value=b.backgroundIntensity,l.material.uniforms.backgroundRotation.value.setFromMatrix4(c0.makeRotationFromEuler(b.backgroundRotation)).transpose(),_.isRenderTargetTexture===!1&&l.material.uniforms.backgroundRotation.value.premultiply(cd),l.material.toneMapped=oe.getTransfer(_.colorSpace)!==ge,(u!==_||h!==_.version||d!==i.toneMapping)&&(l.material.needsUpdate=!0,u=_,h=_.version,d=i.toneMapping),l.layers.enableAll(),v.unshift(l,l.geometry,l.material,0,0,null)):_&&_.isTexture&&(c===void 0&&(c=new he(new ni(2,2),new dn({name:"BackgroundMaterial",uniforms:Ji(zn.background.uniforms),vertexShader:zn.background.vertexShader,fragmentShader:zn.background.fragmentShader,side:wi,depthTest:!1,depthWrite:!1,fog:!1,allowOverride:!1})),c.geometry.deleteAttribute("normal"),Object.defineProperty(c.material,"map",{get:function(){return this.uniforms.t2D.value}}),n.update(c)),c.material.uniforms.t2D.value=_,c.material.uniforms.backgroundIntensity.value=b.backgroundIntensity,c.material.toneMapped=oe.getTransfer(_.colorSpace)!==ge,_.matrixAutoUpdate===!0&&_.updateMatrix(),c.material.uniforms.uvTransform.value.copy(_.matrix),(u!==_||h!==_.version||d!==i.toneMapping)&&(c.material.needsUpdate=!0,u=_,h=_.version,d=i.toneMapping),c.layers.enableAll(),v.unshift(c,c.geometry,c.material,0,0,null))}function g(v,b){v.getRGB($o,wc(i)),e.buffers.color.setClear($o.r,$o.g,$o.b,b,r)}function p(){l!==void 0&&(l.geometry.dispose(),l.material.dispose(),l=void 0),c!==void 0&&(c.geometry.dispose(),c.material.dispose(),c=void 0)}return{getClearColor:function(){return a},setClearColor:function(v,b=1){a.set(v),o=b,g(a,o)},getClearAlpha:function(){return o},setClearAlpha:function(v){o=v,g(a,o)},render:m,addToRenderList:y,dispose:p}}function u0(i,t){let e=i.getParameter(i.MAX_VERTEX_ATTRIBS),n={},s=d(null),r=s,a=!1;function o(k,O,j,D,K){let Q=!1,Y=h(k,D,j,O);r!==Y&&(r=Y,l(r.object)),Q=f(k,D,j,K),Q&&m(k,D,j,K),K!==null&&t.update(K,i.ELEMENT_ARRAY_BUFFER),(Q||a)&&(a=!1,_(k,O,j,D),K!==null&&i.bindBuffer(i.ELEMENT_ARRAY_BUFFER,t.get(K).buffer))}function c(){return i.createVertexArray()}function l(k){return i.bindVertexArray(k)}function u(k){return i.deleteVertexArray(k)}function h(k,O,j,D){let K=D.wireframe===!0,Q=n[O.id];Q===void 0&&(Q={},n[O.id]=Q);let Y=k.isInstancedMesh===!0?k.id:0,J=Q[Y];J===void 0&&(J={},Q[Y]=J);let at=J[j.id];at===void 0&&(at={},J[j.id]=at);let W=at[K];return W===void 0&&(W=d(c()),at[K]=W),W}function d(k){let O=[],j=[],D=[];for(let K=0;K<e;K++)O[K]=0,j[K]=0,D[K]=0;return{geometry:null,program:null,wireframe:!1,newAttributes:O,enabledAttributes:j,attributeDivisors:D,object:k,attributes:{},index:null}}function f(k,O,j,D){let K=r.attributes,Q=O.attributes,Y=0,J=j.getAttributes();for(let at in J)if(J[at].location>=0){let tt=K[at],vt=Q[at];if(vt===void 0&&(at==="instanceMatrix"&&k.instanceMatrix&&(vt=k.instanceMatrix),at==="instanceColor"&&k.instanceColor&&(vt=k.instanceColor)),tt===void 0||tt.attribute!==vt||vt&&tt.data!==vt.data)return!0;Y++}return r.attributesNum!==Y||r.index!==D}function m(k,O,j,D){let K={},Q=O.attributes,Y=0,J=j.getAttributes();for(let at in J)if(J[at].location>=0){let tt=Q[at];tt===void 0&&(at==="instanceMatrix"&&k.instanceMatrix&&(tt=k.instanceMatrix),at==="instanceColor"&&k.instanceColor&&(tt=k.instanceColor));let vt={};vt.attribute=tt,tt&&tt.data&&(vt.data=tt.data),K[at]=vt,Y++}r.attributes=K,r.attributesNum=Y,r.index=D}function y(){let k=r.newAttributes;for(let O=0,j=k.length;O<j;O++)k[O]=0}function g(k){p(k,0)}function p(k,O){let j=r.newAttributes,D=r.enabledAttributes,K=r.attributeDivisors;j[k]=1,D[k]===0&&(i.enableVertexAttribArray(k),D[k]=1),K[k]!==O&&(i.vertexAttribDivisor(k,O),K[k]=O)}function v(){let k=r.newAttributes,O=r.enabledAttributes;for(let j=0,D=O.length;j<D;j++)O[j]!==k[j]&&(i.disableVertexAttribArray(j),O[j]=0)}function b(k,O,j,D,K,Q,Y){Y===!0?i.vertexAttribIPointer(k,O,j,K,Q):i.vertexAttribPointer(k,O,j,D,K,Q)}function _(k,O,j,D){y();let K=D.attributes,Q=j.getAttributes(),Y=O.defaultAttributeValues;for(let J in Q){let at=Q[J];if(at.location>=0){let W=K[J];if(W===void 0&&(J==="instanceMatrix"&&k.instanceMatrix&&(W=k.instanceMatrix),J==="instanceColor"&&k.instanceColor&&(W=k.instanceColor)),W!==void 0){let tt=W.normalized,vt=W.itemSize,wt=t.get(W);if(wt===void 0)continue;let st=wt.buffer,U=wt.type,et=wt.bytesPerElement,V=U===i.INT||U===i.UNSIGNED_INT||W.gpuType===co;if(W.isInterleavedBufferAttribute){let N=W.data,L=N.isInterleavedBuffer===!0?N.stride:N.itemSize,I=W.offset;if(N.isInstancedInterleavedBuffer||N.isInstancedBufferAttribute){for(let G=0;G<at.locationSize;G++)p(at.location+G,N.meshPerAttribute);k.isInstancedMesh!==!0&&D._maxInstanceCount===void 0&&(D._maxInstanceCount=N.meshPerAttribute*N.count)}else for(let G=0;G<at.locationSize;G++)g(at.location+G);i.bindBuffer(i.ARRAY_BUFFER,st);for(let G=0;G<at.locationSize;G++)b(at.location+G,vt/at.locationSize,U,tt,L*et,(I+vt/at.locationSize*G)*et,V)}else{if(W.isInstancedBufferAttribute){for(let N=0;N<at.locationSize;N++)p(at.location+N,W.meshPerAttribute);k.isInstancedMesh!==!0&&D._maxInstanceCount===void 0&&(D._maxInstanceCount=W.meshPerAttribute*W.count)}else for(let N=0;N<at.locationSize;N++)g(at.location+N);i.bindBuffer(i.ARRAY_BUFFER,st);for(let N=0;N<at.locationSize;N++)b(at.location+N,vt/at.locationSize,U,tt,vt*et,vt/at.locationSize*N*et,V)}}else if(Y!==void 0){let tt=Y[J];if(tt!==void 0)switch(tt.length){case 2:i.vertexAttrib2fv(at.location,tt);break;case 3:i.vertexAttrib3fv(at.location,tt);break;case 4:i.vertexAttrib4fv(at.location,tt);break;default:i.vertexAttrib1fv(at.location,tt)}}}}v()}function T(){P();for(let k in n){let O=n[k];for(let j in O){let D=O[j];for(let K in D){let Q=D[K];for(let Y in Q)u(Q[Y].object),delete Q[Y];delete D[K]}}delete n[k]}}function w(k){if(n[k.id]===void 0)return;let O=n[k.id];for(let j in O){let D=O[j];for(let K in D){let Q=D[K];for(let Y in Q)u(Q[Y].object),delete Q[Y];delete D[K]}}delete n[k.id]}function x(k){for(let O in n){let j=n[O];for(let D in j){let K=j[D];if(K[k.id]===void 0)continue;let Q=K[k.id];for(let Y in Q)u(Q[Y].object),delete Q[Y];delete K[k.id]}}}function M(k){for(let O in n){let j=n[O],D=k.isInstancedMesh===!0?k.id:0,K=j[D];if(K!==void 0){for(let Q in K){let Y=K[Q];for(let J in Y)u(Y[J].object),delete Y[J];delete K[Q]}delete j[D],Object.keys(j).length===0&&delete n[O]}}}function P(){C(),a=!0,r!==s&&(r=s,l(r.object))}function C(){s.geometry=null,s.program=null,s.wireframe=!1}return{setup:o,reset:P,resetDefaultState:C,dispose:T,releaseStatesOfGeometry:w,releaseStatesOfObject:M,releaseStatesOfProgram:x,initAttributes:y,enableAttribute:g,disableUnusedAttributes:v}}function d0(i,t,e){let n;function s(c){n=c}function r(c,l){i.drawArrays(n,c,l),e.update(l,n,1)}function a(c,l,u){u!==0&&(i.drawArraysInstanced(n,c,l,u),e.update(l,n,u))}function o(c,l,u){if(u===0)return;t.get("WEBGL_multi_draw").multiDrawArraysWEBGL(n,c,0,l,0,u);let d=0;for(let f=0;f<u;f++)d+=l[f];e.update(d,n,1)}this.setMode=s,this.render=r,this.renderInstances=a,this.renderMultiDraw=o}function f0(i,t,e,n){let s;function r(){if(s!==void 0)return s;if(t.has("EXT_texture_filter_anisotropic")===!0){let x=t.get("EXT_texture_filter_anisotropic");s=i.getParameter(x.MAX_TEXTURE_MAX_ANISOTROPY_EXT)}else s=0;return s}function a(x){return!(x!==Dn&&n.convert(x)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_FORMAT))}function o(x){let M=x===Ln&&(t.has("EXT_color_buffer_half_float")||t.has("EXT_color_buffer_float"));return!(x!==nn&&x!==xn&&!M&&n.convert(x)!==i.getParameter(i.IMPLEMENTATION_COLOR_READ_TYPE))}function c(x){if(x==="highp"){if(i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.HIGH_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.HIGH_FLOAT).precision>0)return"highp";x="mediump"}return x==="mediump"&&i.getShaderPrecisionFormat(i.VERTEX_SHADER,i.MEDIUM_FLOAT).precision>0&&i.getShaderPrecisionFormat(i.FRAGMENT_SHADER,i.MEDIUM_FLOAT).precision>0?"mediump":"lowp"}let l=e.precision!==void 0?e.precision:"highp",u=c(l);u!==l&&(qt("WebGLRenderer:",l,"not supported, using",u,"instead."),l=u);let h=e.reversedDepthBuffer===!0&&t.has("EXT_clip_control"),d=e.logarithmicDepthBuffer===!0&&h===!1;e.reversedDepthBuffer===!0&&h===!1&&qt("WebGLRenderer: Unable to use reversed depth buffer due to missing EXT_clip_control extension. Fallback to default depth buffer."),e.logarithmicDepthBuffer===!0&&h===!0&&qt("WebGLRenderer: Logarithmic depth buffer is ignored when reversed depth buffer is enabled.");let f=i.getParameter(i.MAX_TEXTURE_IMAGE_UNITS),m=i.getParameter(i.MAX_VERTEX_TEXTURE_IMAGE_UNITS),y=i.getParameter(i.MAX_TEXTURE_SIZE),g=i.getParameter(i.MAX_CUBE_MAP_TEXTURE_SIZE),p=i.getParameter(i.MAX_VERTEX_ATTRIBS),v=i.getParameter(i.MAX_VERTEX_UNIFORM_VECTORS),b=i.getParameter(i.MAX_VARYING_VECTORS),_=i.getParameter(i.MAX_FRAGMENT_UNIFORM_VECTORS),T=i.getParameter(i.MAX_SAMPLES),w=i.getParameter(i.SAMPLES);return{isWebGL2:!0,getMaxAnisotropy:r,getMaxPrecision:c,textureFormatReadable:a,textureTypeReadable:o,precision:l,logarithmicDepthBuffer:d,reversedDepthBuffer:h,maxTextures:f,maxVertexTextures:m,maxTextureSize:y,maxCubemapSize:g,maxAttributes:p,maxVertexUniforms:v,maxVaryings:b,maxFragmentUniforms:_,maxSamples:T,samples:w}}function p0(i){let t=this,e=null,n=0,s=!1,r=!1,a=new on,o=new Xt,c={value:null,needsUpdate:!1};this.uniform=c,this.numPlanes=0,this.numIntersection=0,this.init=function(h,d){let f=h.length!==0||d||n!==0||s;return s=d,n=h.length,f},this.beginShadows=function(){r=!0,u(null)},this.endShadows=function(){r=!1},this.setGlobalState=function(h,d){e=u(h,d,0)},this.setState=function(h,d,f){let m=h.clippingPlanes,y=h.clipIntersection,g=h.clipShadows,p=i.get(h);if(!s||m===null||m.length===0||r&&!g)r?u(null):l();else{let v=r?0:n,b=v*4,_=p.clippingState||null;c.value=_,_=u(m,d,b,f);for(let T=0;T!==b;++T)_[T]=e[T];p.clippingState=_,this.numIntersection=y?this.numPlanes:0,this.numPlanes+=v}};function l(){c.value!==e&&(c.value=e,c.needsUpdate=n>0),t.numPlanes=n,t.numIntersection=0}function u(h,d,f,m){let y=h!==null?h.length:0,g=null;if(y!==0){if(g=c.value,m!==!0||g===null){let p=f+y*4,v=d.matrixWorldInverse;o.getNormalMatrix(v),(g===null||g.length<p)&&(g=new Float32Array(p));for(let b=0,_=f;b!==y;++b,_+=4)a.copy(h[b]).applyMatrix4(v,o),a.normal.toArray(g,_),g[_+3]=a.constant}c.value=g,c.needsUpdate=!0}return t.numPlanes=y,t.numIntersection=0,g}}var Hr=class extends cn{constructor(t=1,e={}){super(t,t,e),this.isWebGLCubeRenderTarget=!0;let n={width:t,height:t,depth:1},s=[n,n,n,n,n,n];this.texture=new fr(s),this._setTextureOptions(e),this.texture.isRenderTargetTexture=!0}fromEquirectangularTexture(t,e){this.texture.type=e.type,this.texture.colorSpace=e.colorSpace,this.texture.generateMipmaps=e.generateMipmaps,this.texture.minFilter=e.minFilter,this.texture.magFilter=e.magFilter;let n={uniforms:{tEquirect:{value:null}},vertexShader:`

				varying vec3 vWorldDirection;

				vec3 transformDirection( in vec3 dir, in mat4 matrix ) {

					return normalize( ( matrix * vec4( dir, 0.0 ) ).xyz );

				}

				void main() {

					vWorldDirection = transformDirection( position, modelMatrix );

					#include <begin_vertex>
					#include <project_vertex>

				}
			`,fragmentShader:`

				uniform sampler2D tEquirect;

				varying vec3 vWorldDirection;

				#include <common>

				void main() {

					vec3 direction = normalize( vWorldDirection );

					vec2 sampleUV = equirectUv( direction );

					gl_FragColor = texture2D( tEquirect, sampleUV );

				}
			`},s=new Ce(5,5,5),r=new dn({name:"CubemapFromEquirect",uniforms:Ji(n.uniforms),vertexShader:n.vertexShader,fragmentShader:n.fragmentShader,side:We,blending:Xi});r.uniforms.tEquirect.value=e;let a=new he(s,r),o=e.minFilter;return e.minFilter===si&&(e.minFilter=je),new Rs(1,10,this).update(t,a),e.minFilter=o,a.geometry.dispose(),a.material.dispose(),this}clear(t,e=!0,n=!0,s=!0){let r=t.getRenderTarget();for(let a=0;a<6;a++)t.setRenderTarget(this,a),t.clear(e,n,s);t.setRenderTarget(r)}},Gu=256,m0=3,g0=20,hd=256,ud=16,x0=3,_0=new F,y0=new Vt,Jo=class i{constructor(t){this._renderer=t,this._cubeSize=0,this._sourceTarget=null,this._cubeCamera=new Rs(1,10,null),this._boxMesh=new he(new Ce(5,5,5),null),this._cubemapMaterial=null,this._equirectMaterial=null,this._blurMaterial=null,this._ggxMaterial=null,this._integrationMaterial=null}fromScene(t,e=0,n=.1,s=100,r={}){let{size:a=256,position:o=_0}=r,c=this._renderer;this._setSize(a);let l=this._allocateTarget(),u=this._getSourceTarget(!0);e>0&&(u.texture.mipmapsAutoUpdate=!1);let h=c.autoClear,d=c.autoClearColor,f=c.autoClearDepth,m=c.autoClearStencil,y=t.background;c.autoClear=!0,c.autoClearColor=!0,c.autoClearDepth=!0,c.autoClearStencil=!0,y===null&&(t.background=c.getClearColor(y0));let g=new Rs(n,s,u);return g.position.copy(o),g.update(c,t),c.autoClear=h,c.autoClearColor=d,c.autoClearDepth=f,c.autoClearStencil=m,t.background=y,e>0&&(u.texture.mipmapsAutoUpdate=!0,this._blur(l,e)),this._applyPMREM(l),l}fromEquirectangular(t,e=null){return this._fromTexture(t,e)}fromCubemap(t,e=null){return this._fromTexture(t,e)}compileCubemapShader(){this._cubemapMaterial===null&&(this._cubemapMaterial=Wu(),this._compileMaterial(this._cubemapMaterial))}compileEquirectangularShader(){this._equirectMaterial===null&&(this._equirectMaterial=Hu(),this._compileMaterial(this._equirectMaterial))}dispose(){this._sourceTarget!==null&&this._sourceTarget.dispose(),this._cubemapMaterial!==null&&this._cubemapMaterial.dispose(),this._equirectMaterial!==null&&this._equirectMaterial.dispose(),this._blurMaterial!==null&&this._blurMaterial.dispose(),this._ggxMaterial!==null&&this._ggxMaterial.dispose(),this._integrationMaterial!==null&&this._integrationMaterial.dispose(),this._boxMesh.geometry.dispose()}static lodToRoughness(t,e){return e>0?1-Math.sqrt(1-t/e):0}_setSize(t){this._cubeSize=Math.max(Gu,bc(t))}_fromTexture(t,e){t.mapping===Yi||t.mapping===Ns?this._setSize(t.image.length===0?Gu:t.image[0].width||t.image[0].image.width):this._setSize(t.image.width/4);let n=e||this._allocateTarget();return this._textureToCubemap(t),this._applyPMREM(n),n}_allocateTarget(){let t=this._cubeSize,e=Vu(t,!1,!1),n=Math.log2(t)-m0;for(let s=0;s<=n;s++)e.texture.mipmaps.push({width:t>>s,height:t>>s});return e.texture.name="PMREM",e.texture.isPMREMTexture=!0,e}_getSourceTarget(t=!1){let e=this._cubeSize,n=this._sourceTarget;return(n===null||n.width!==e||t&&n.depthBuffer===!1)&&(n!==null&&n.dispose(),this._sourceTarget=Vu(e,!0,t)),this._sourceTarget}_compileMaterial(t){this._boxMesh.material=t,this._renderer.compile(this._boxMesh,this._cubeCamera.children[0])}_renderCube(t,e,n){let s=this._boxMesh,r=this._cubeCamera;s.material=n,r.renderTarget=t,r.activeMipmapLevel=e;let a=t.width>>e;t.viewport.set(0,0,a,a),r.update(this._renderer,s),t.viewport.set(0,0,t.width,t.height)}_textureToCubemap(t){let e;t.mapping===Yi||t.mapping===Ns?(this._cubemapMaterial===null&&(this._cubemapMaterial=Wu()),e=this._cubemapMaterial,e.uniforms.flipEnvMap.value=t.isRenderTargetTexture===!1?-1:1):(this._equirectMaterial===null&&(this._equirectMaterial=Hu()),e=this._equirectMaterial),e.uniforms.envMap.value=t,this._renderCube(this._getSourceTarget(),0,e)}_applyPMREM(t){this._ggxMaterial===null&&(this._ggxMaterial=v0(),this._integrationMaterial=b0());let e=this._cubeSize,n=t.texture.mipmaps.length-1,s=this._ggxMaterial.uniforms;s.envMap.value=this._sourceTarget.texture;let r=this._integrationMaterial.uniforms;r.envMap.value=this._sourceTarget.texture,r.sourceLod.value=Math.log2(e/ud);for(let a=0;a<=n;a++){let o=i.lodToRoughness(a,n);if(a>n-x0)r.roughness.value=o,this._renderCube(t,a,this._integrationMaterial);else{let c=o>0?Math.log2(e)+.5*Math.log2(6/(hd*Math.pow(o,4)))+.5:0;s.roughness.value=o,s.lodBias.value=c,this._renderCube(t,a,this._ggxMaterial)}}}_blur(t,e){this._blurMaterial===null&&(this._blurMaterial=M0());let n=this._blurMaterial,s=n.uniforms,r=this._sourceTarget;s.sigma.value=Math.min(e,Math.PI)/Math.SQRT2,s.envMap.value=r.texture,this._renderCube(t,0,n),s.envMap.value=t.texture,this._renderCube(r,0,n)}};function Vu(i,t,e){return new Hr(i,{minFilter:si,generateMipmaps:t,type:Ln,colorSpace:ir,depthBuffer:e})}function Wr(i,t,e){return new dn({name:i,uniforms:t,vertexShader:ie.cube_vert,fragmentShader:e,side:We,blending:Xi,depthTest:!1,depthWrite:!1})}function v0(){return Wr("PMREMGGXConvolution",{envMap:{value:null},roughness:{value:0},lodBias:{value:0}},`

		#define GGX_SAMPLES ${hd}u

		varying vec3 vWorldDirection;

		uniform samplerCube envMap;
		uniform float roughness;
		uniform float lodBias;

		#include <common>

		// Van der Corput radical inverse
		float radicalInverse_VdC( uint bits ) {

			bits = ( bits << 16u ) | ( bits >> 16u );
			bits = ( ( bits & 0x55555555u ) << 1u ) | ( ( bits & 0xAAAAAAAAu ) >> 1u );
			bits = ( ( bits & 0x33333333u ) << 2u ) | ( ( bits & 0xCCCCCCCCu ) >> 2u );
			bits = ( ( bits & 0x0F0F0F0Fu ) << 4u ) | ( ( bits & 0xF0F0F0F0u ) >> 4u );
			bits = ( ( bits & 0x00FF00FFu ) << 8u ) | ( ( bits & 0xFF00FF00u ) >> 8u );
			return float( bits ) * 2.3283064365386963e-10; // / 0x100000000

		}

		// Hammersley sequence
		vec2 hammersley( uint i, uint N ) {

			return vec2( float( i ) / float( N ), radicalInverse_VdC( i ) );

		}

		void main() {

			vec3 N = normalize( vWorldDirection );

			// For very low roughness, just sample the environment directly
			if ( roughness < 0.001 ) {

				gl_FragColor = vec4( textureLod( envMap, N, 0.0 ).rgb, 1.0 );
				return;

			}

			float alpha = roughness * roughness;
			float alpha2 = alpha * alpha;

			// Tangent space basis for VNDF sampling
			vec3 up = abs( N.z ) < 0.999 ? vec3( 0.0, 0.0, 1.0 ) : vec3( 1.0, 0.0, 0.0 );
			vec3 tangent = normalize( cross( up, N ) );
			vec3 bitangent = cross( N, tangent );

			vec3 prefilteredColor = vec3( 0.0 );
			float totalWeight = 0.0;

			for ( uint i = 0u; i < GGX_SAMPLES; i ++ ) {

				vec2 Xi = hammersley( i, GGX_SAMPLES );

				// With V = N, sample the reflected direction directly.
				float invQ = 1.0 / ( 1.0 - Xi.x + alpha2 * Xi.x );
				float NdotL = ( 1.0 - Xi.x - alpha2 * Xi.x ) * invQ;

				if ( NdotL > 0.0 ) {

					float phi = 2.0 * PI * Xi.y;
					float sinTheta = 2.0 * alpha * sqrt( Xi.x * ( 1.0 - Xi.x ) ) * invQ;
					vec3 L = N * NdotL + ( tangent * cos( phi ) + bitangent * sin( phi ) ) * sinTheta;

					// Match the source mip to the sample's solid angle. The terms
					// independent of the sample are precomputed in lodBias.
					float d = alpha2 * invQ;
					float lod = max( log2( d ) + lodBias, 0.0 );

					// Weight by NdotL for the split-sum approximation
					prefilteredColor += textureLod( envMap, L, lod ).rgb * NdotL;
					totalWeight += NdotL;

				}

			}

			gl_FragColor = vec4( prefilteredColor / totalWeight, 1.0 );

		}
	`)}function b0(){return Wr("PMREMGGXIntegration",{envMap:{value:null},roughness:{value:0},sourceLod:{value:0},sourceSize:{value:ud}},`

		varying vec3 vWorldDirection;

		uniform samplerCube envMap;
		uniform float roughness;
		uniform float sourceLod;
		uniform int sourceSize;

		void main() {

			vec3 N = normalize( vWorldDirection );

			float alpha = roughness * roughness;
			float alpha2 = alpha * alpha;

			float texelSize = 2.0 / float( sourceSize );

			vec3 prefilteredColor = vec3( 0.0 );
			float totalWeight = 0.0;

			// Pair opposite texels: only the one in N's hemisphere contributes.
			for ( int face = 0; face < 3; face ++ ) {

				for ( int y = 0; y < sourceSize; y ++ ) {

					for ( int x = 0; x < sourceSize; x ++ ) {

						vec2 uv = ( vec2( x, y ) + 0.5 ) * texelSize - 1.0;
						vec3 texelDirection = face == 0 ? vec3( 1.0, uv ) : ( face == 1 ? vec3( uv.x, 1.0, uv.y ) : vec3( uv, 1.0 ) );

						float invDistance = inversesqrt( 1.0 + dot( uv, uv ) );
						float NdotL = dot( N, texelDirection );
						texelDirection *= NdotL < 0.0 ? - 1.0 : 1.0;
						NdotL = abs( NdotL ) * invDistance;

						// With V = N, NdotH squared is ( 1 + NdotL ) / 2. Common factors
						// in the GGX distribution and texel solid angle cancel when normalized.
						float d = 1.0 + alpha2 + ( alpha2 - 1.0 ) * NdotL;
						float weight = NdotL * invDistance * invDistance * invDistance / ( d * d );

						prefilteredColor += textureLod( envMap, texelDirection, sourceLod ).rgb * weight;
						totalWeight += weight;

					}

				}

			}

			gl_FragColor = vec4( prefilteredColor / totalWeight, 1.0 );

		}
	`)}function M0(){return Wr("PMREMSphericalGaussianBlur",{envMap:{value:null},sigma:{value:0}},`

		#define SAMPLES ${g0}
		#define GOLDEN_ANGLE 2.39996322973

		varying vec3 vWorldDirection;

		uniform samplerCube envMap;
		uniform float sigma;

		#include <common>

		void main() {

			vec3 outputDirection = normalize( vWorldDirection );

			vec3 up = abs( outputDirection.z ) < 0.999 ? vec3( 0.0, 0.0, 1.0 ) : vec3( 1.0, 0.0, 0.0 );
			vec3 tangent = normalize( cross( up, outputDirection ) );
			vec3 bitangent = cross( outputDirection, tangent );

			// Truncate the kernel at three standard deviations or at the antipode.
			float thetaMax = min( 3.0 * sigma, PI );
			float truncation = 1.0 - exp( - 0.5 * thetaMax * thetaMax / ( sigma * sigma ) );

			vec3 accumColor = vec3( 0.0 );
			float accumWeight = 0.0;

			for ( int i = 0; i < SAMPLES; i ++ ) {

				// Stratified inverse-CDF sampling of the Gaussian, placed on a golden-angle spiral.
				float stratum = ( float( i ) + 0.5 ) / float( SAMPLES );
				float theta = sigma * sqrt( - 2.0 * log( 1.0 - stratum * truncation ) );
				float phi = float( i ) * GOLDEN_ANGLE;

				vec3 offset = cos( phi ) * tangent + sin( phi ) * bitangent;
				vec3 sampleDirection = cos( theta ) * outputDirection + sin( theta ) * offset;

				// Correct the planar sample density to solid angle.
				float weight = sin( theta ) / theta;

				accumColor += weight * textureLod( envMap, sampleDirection, 0.0 ).rgb;
				accumWeight += weight;

			}

			gl_FragColor = vec4( accumColor / accumWeight, 1.0 );

		}
	`)}function Hu(){return Wr("PMREMEquirectangularToCubemap",{envMap:{value:null}},`

		varying vec3 vWorldDirection;

		uniform sampler2D envMap;

		#include <common>

		void main() {

			// Average four subpixel samples to preserve small, bright features.
			vec3 direction = normalize( vWorldDirection );
			vec3 dx = dFdx( direction ) * 0.25;
			vec3 dy = dFdy( direction ) * 0.25;

			vec3 color = textureLod( envMap, equirectUv( normalize( direction - dx - dy ) ), 0.0 ).rgb;
			color += textureLod( envMap, equirectUv( normalize( direction + dx - dy ) ), 0.0 ).rgb;
			color += textureLod( envMap, equirectUv( normalize( direction - dx + dy ) ), 0.0 ).rgb;
			color += textureLod( envMap, equirectUv( normalize( direction + dx + dy ) ), 0.0 ).rgb;
			gl_FragColor = vec4( color * 0.25, 1.0 );

		}
	`)}function Wu(){return Wr("PMREMCubemapToCubemap",{envMap:{value:null},flipEnvMap:{value:-1}},`

		varying vec3 vWorldDirection;

		uniform samplerCube envMap;
		uniform float flipEnvMap;

		void main() {

			gl_FragColor = vec4( textureCube( envMap, vec3( flipEnvMap * vWorldDirection.x, vWorldDirection.yz ) ).rgb, 1.0 );

		}
	`)}function S0(i){let t=new WeakMap,e=new WeakMap,n=null;function s(d,f=!1){return d==null?null:f?a(d):r(d)}function r(d){if(d&&d.isTexture){let f=d.mapping;if(f===ao||f===oo)if(t.has(d)){let m=t.get(d).texture;return o(m,d.mapping)}else{let m=d.image;if(m&&m.height>0){let y=new Hr(m.height);return y.fromEquirectangularTexture(i,d),t.set(d,y),d.addEventListener("dispose",l),o(y.texture,d.mapping)}else return null}}return d}function a(d){if(d&&d.isTexture&&d.isPMREMTexture!==!0){let f=d.mapping,m=f===ao||f===oo,y=f===Yi||f===Ns;if(m||y){let g=e.get(d),p=g!==void 0?g.texture.pmremVersion:0;if(d.isRenderTargetTexture&&d.pmremVersion!==p)return n===null&&(n=new Jo(i)),g=m?n.fromEquirectangular(d,g):n.fromCubemap(d,g),g.texture.pmremVersion=d.pmremVersion,e.set(d,g),g.texture;if(g!==void 0)return g.texture;{let v=d.image;return m&&v&&v.height>0||y&&v&&c(v)?(n===null&&(n=new Jo(i)),g=m?n.fromEquirectangular(d):n.fromCubemap(d),g.texture.pmremVersion=d.pmremVersion,e.set(d,g),d.addEventListener("dispose",u),g.texture):null}}}return d}function o(d,f){return f===ao?d.mapping=Yi:f===oo&&(d.mapping=Ns),d}function c(d){let f=0,m=6;for(let y=0;y<m;y++)d[y]!==void 0&&f++;return f===m}function l(d){let f=d.target;f.removeEventListener("dispose",l);let m=t.get(f);m!==void 0&&(t.delete(f),m.dispose())}function u(d){let f=d.target;f.removeEventListener("dispose",u);let m=e.get(f);m!==void 0&&(e.delete(f),m.dispose())}function h(){t=new WeakMap,e=new WeakMap,n!==null&&(n.dispose(),n=null)}return{get:s,dispose:h}}function w0(i){let t={};function e(n){if(t[n]!==void 0)return t[n];let s=i.getExtension(n);return t[n]=s,s}return{has:function(n){return e(n)!==null},init:function(){e("EXT_color_buffer_float"),e("WEBGL_clip_cull_distance"),e("OES_texture_float_linear"),e("EXT_color_buffer_half_float"),e("WEBGL_multisampled_render_to_texture"),e("WEBGL_render_shared_exponent")},get:function(n){let s=e(n);return s===null&&Ks("WebGLRenderer: "+n+" extension not supported."),s}}}function T0(i,t,e,n){let s={},r=new WeakMap,a=new FinalizationRegistry(m=>delete s[m]);function o(m){c(m.target)}function c(m){m.index!==null&&t.remove(m.index);for(let g in m.attributes)t.remove(m.attributes[g]);m.removeEventListener("dispose",o),delete s[m.id],a.unregister(m);let y=r.get(m);y&&(t.remove(y),r.delete(m)),n.releaseStatesOfGeometry(m),m.isInstancedBufferGeometry===!0&&delete m._maxInstanceCount,e.memory.geometries--}function l(m,y){return s[y.id]!==void 0||(y.addEventListener("dispose",o),s[y.id]=new WeakRef(y),a.register(y,y.id,y),e.memory.geometries++),y}function u(m){let y=m.attributes;for(let g in y)t.update(y[g],i.ARRAY_BUFFER)}function h(m){let y=m.index,g=m.attributes.position,p=0;if(g===void 0)return;let v=y!==null?y.array.length:g.array.length/3-1,b=g.count>=65535?cr:lr,_=new b(Math.ceil(v/3)*6,1),T=_.array;if(y!==null){let x=y.array;p=y.version;for(let M=0,P=0;M<v;M+=3){let C=x[M+0],k=x[M+1],O=x[M+2];T[P++]=C,T[P++]=k,T[P++]=k,T[P++]=O,T[P++]=O,T[P++]=C}}else{p=g.version;for(let x=0,M=0;x<v;x+=3){let P=x+0,C=x+1,k=x+2;T[M++]=P,T[M++]=C,T[M++]=C,T[M++]=k,T[M++]=k,T[M++]=P}}_.version=p;let w=r.get(m);w&&t.remove(w),r.set(m,_)}function d(m){let y=r.get(m);if(y){let g=m.index;g!==null&&y.version<g.version&&h(m)}else h(m);return r.get(m)}function f(){for(let m in s){let y=s[m].deref();y!==void 0&&c(y)}}return{get:l,update:u,getWireframeAttribute:d,dispose:f}}function A0(i,t,e){let n;function s(h){n=h}let r,a;function o(h){r=h.type,a=h.bytesPerElement}function c(h,d){i.drawElements(n,d,r,h*a),e.update(d,n,1)}function l(h,d,f){f!==0&&(i.drawElementsInstanced(n,d,r,h*a,f),e.update(d,n,f))}function u(h,d,f){if(f===0)return;t.get("WEBGL_multi_draw").multiDrawElementsWEBGL(n,d,0,r,h,0,f);let y=0;for(let g=0;g<f;g++)y+=d[g];e.update(y,n,1)}this.setMode=s,this.setIndex=o,this.render=c,this.renderInstances=l,this.renderMultiDraw=u}function E0(i){let t={geometries:0,textures:0},e={frame:0,calls:0,triangles:0,points:0,lines:0};function n(r,a,o){switch(e.calls++,a){case i.TRIANGLES:e.triangles+=o*(r/3);break;case i.LINES:e.lines+=o*(r/2);break;case i.LINE_STRIP:e.lines+=o*(r-1);break;case i.LINE_LOOP:e.lines+=o*r;break;case i.POINTS:e.points+=o*r;break;default:Zt("WebGLInfo: Unknown draw mode:",a);break}}function s(){e.calls=0,e.triangles=0,e.points=0,e.lines=0}return{memory:t,render:e,programs:null,autoReset:!0,reset:s,update:n}}function C0(i,t,e){let n=new WeakMap,s=new Se;function r(a,o,c){let l=a.morphTargetInfluences,u=o.morphAttributes.position||o.morphAttributes.normal||o.morphAttributes.color,h=u!==void 0?u.length:0,d=n.get(o);if(d===void 0||d.count!==h){let P=function(){x.dispose(),n.delete(o),o.removeEventListener("dispose",P)};d!==void 0&&d.texture.dispose();let f=o.morphAttributes.position!==void 0,m=o.morphAttributes.normal!==void 0,y=o.morphAttributes.color!==void 0,g=o.morphAttributes.position||[],p=o.morphAttributes.normal||[],v=o.morphAttributes.color||[],b=0;f===!0&&(b=1),m===!0&&(b=2),y===!0&&(b=3);let _=o.attributes.position.count*b,T=1;_>t.maxTextureSize&&(T=Math.ceil(_/t.maxTextureSize),_=t.maxTextureSize);let w=new Float32Array(_*T*4*h),x=new ar(w,_,T,h);x.type=xn,x.needsUpdate=!0;let M=b*4;for(let C=0;C<h;C++){let k=g[C],O=p[C],j=v[C],D=_*T*4*C;for(let K=0;K<k.count;K++){let Q=K*M;f===!0&&(s.fromBufferAttribute(k,K),w[D+Q+0]=s.x,w[D+Q+1]=s.y,w[D+Q+2]=s.z,w[D+Q+3]=0),m===!0&&(s.fromBufferAttribute(O,K),w[D+Q+4]=s.x,w[D+Q+5]=s.y,w[D+Q+6]=s.z,w[D+Q+7]=0),y===!0&&(s.fromBufferAttribute(j,K),w[D+Q+8]=s.x,w[D+Q+9]=s.y,w[D+Q+10]=s.z,w[D+Q+11]=j.itemSize===4?s.w:1)}}d={count:h,texture:x,size:new ft(_,T)},n.set(o,d),o.addEventListener("dispose",P)}if(a.isInstancedMesh===!0&&a.morphTexture!==null)c.getUniforms().setValue(i,"morphTexture",a.morphTexture,e);else{let f=0;for(let y=0;y<l.length;y++)f+=l[y];let m=o.morphTargetsRelative?1:1-f;c.getUniforms().setValue(i,"morphTargetBaseInfluence",m),c.getUniforms().setValue(i,"morphTargetInfluences",l)}c.getUniforms().setValue(i,"morphTargetsTexture",d.texture,e),c.getUniforms().setValue(i,"morphTargetsTextureSize",d.size)}return{update:r}}function R0(i,t,e,n,s){let r=new WeakMap;function a(l){let u=s.render.frame,h=l.geometry,d=t.get(l,h);if(r.get(d)!==u&&(t.update(d),r.set(d,u)),l.isInstancedMesh&&(l.hasEventListener("dispose",c)===!1&&l.addEventListener("dispose",c),r.get(l)!==u&&(e.update(l.instanceMatrix,i.ARRAY_BUFFER),l.instanceColor!==null&&e.update(l.instanceColor,i.ARRAY_BUFFER),r.set(l,u))),l.isSkinnedMesh){let f=l.skeleton;r.get(f)!==u&&(f.update(),r.set(f,u))}return d}function o(){r=new WeakMap}function c(l){let u=l.target;u.removeEventListener("dispose",c),n.releaseStatesOfObject(u),e.remove(u.instanceMatrix),u.instanceColor!==null&&e.remove(u.instanceColor)}return{update:a,dispose:o}}var P0={[nc]:"LINEAR_TONE_MAPPING",[ic]:"REINHARD_TONE_MAPPING",[sc]:"CINEON_TONE_MAPPING",[Pn]:"ACES_FILMIC_TONE_MAPPING",[ac]:"AGX_TONE_MAPPING",[oc]:"NEUTRAL_TONE_MAPPING",[rc]:"CUSTOM_TONE_MAPPING"};function I0(i,t,e,n,s,r){let a=new cn(t,e,{type:i,depthBuffer:s,stencilBuffer:r,samples:n?4:0,storeMultisampledDepthBuffer:!1,storeMultisampledStencilBuffer:!1,resolveDepthBuffer:!1,resolveStencilBuffer:!1}),o=null,c=null,l=null,u=new de;u.setAttribute("position",new Jt([-1,3,0,-1,-1,0,3,-1,0],3)),u.setAttribute("uv",new Jt([0,2,0,0,2,0],2));let h=new qa({uniforms:{tDiffuse:{value:null}},vertexShader:`
			precision highp float;

			uniform mat4 modelViewMatrix;
			uniform mat4 projectionMatrix;

			attribute vec3 position;
			attribute vec2 uv;

			varying vec2 vUv;

			void main() {
				vUv = uv;
				gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
			}`,fragmentShader:`
			precision highp float;

			uniform sampler2D tDiffuse;

			varying vec2 vUv;

			#include <tonemapping_pars_fragment>
			#include <colorspace_pars_fragment>

			void main() {
				gl_FragColor = texture2D( tDiffuse, vUv );

				#ifdef LINEAR_TONE_MAPPING
					gl_FragColor.rgb = LinearToneMapping( gl_FragColor.rgb );
				#elif defined( REINHARD_TONE_MAPPING )
					gl_FragColor.rgb = ReinhardToneMapping( gl_FragColor.rgb );
				#elif defined( CINEON_TONE_MAPPING )
					gl_FragColor.rgb = CineonToneMapping( gl_FragColor.rgb );
				#elif defined( ACES_FILMIC_TONE_MAPPING )
					gl_FragColor.rgb = ACESFilmicToneMapping( gl_FragColor.rgb );
				#elif defined( AGX_TONE_MAPPING )
					gl_FragColor.rgb = AgXToneMapping( gl_FragColor.rgb );
				#elif defined( NEUTRAL_TONE_MAPPING )
					gl_FragColor.rgb = NeutralToneMapping( gl_FragColor.rgb );
				#elif defined( CUSTOM_TONE_MAPPING )
					gl_FragColor.rgb = CustomToneMapping( gl_FragColor.rgb );
				#endif

				#ifdef SRGB_TRANSFER
					gl_FragColor = sRGBTransferOETF( gl_FragColor );
				#endif
			}`,depthTest:!1,depthWrite:!1}),d=new he(u,h),f=new En(-1,1,1,-1,0,1),m=null,y=null,g=!1,p,v=null,b=[],_=!1;this.setSize=function(T,w){a.setSize(T,w),o!==null&&o.setSize(T,w),c!==null&&c.setSize(T,w);for(let x=0;x<b.length;x++){let M=b[x];M.setSize&&M.setSize(T,w)}},this.setEffects=function(T){b=T,_=b.length>0&&b[0].isRenderPass===!0;let w=a.width,x=a.height;b.length>0&&o===null&&(o=new cn(w,x,{type:Ln,depthBuffer:!1,stencilBuffer:!1}),c=new cn(w,x,{type:Ln,depthBuffer:!1,stencilBuffer:!1}));let M=s===!0&&b.some(P=>P.needsDepthTexture===!0);M!==(a.depthTexture!==null)&&(M===!0&&l===null&&(l=new ei(w,x),r===!0&&(l.format=ri,l.type=$i)),a.dispose(),a.depthTexture=M?l:null,a.resolveDepthBuffer=M,a.storeMultisampledDepthBuffer=M);for(let P=0;P<b.length;P++){let C=b[P];C.setSize&&C.setSize(w,x),M&&C.needsDepthTexture===!0&&C.setDepthTexture&&C.setDepthTexture(l)}},this.begin=function(T,w){if(g||T.toneMapping===Bn&&b.length===0)return!1;if(v=w,w!==null){let x=w.width,M=w.height;(a.width!==x||a.height!==M)&&this.setSize(x,M)}return _===!1&&T.setRenderTarget(a),p=T.toneMapping,T.toneMapping=Bn,!0},this.hasRenderPass=function(){return _},this.end=function(T,w){T.toneMapping=p,g=!0;let x=a,M=o;for(let P=0;P<b.length;P++){let C=b[P];C.enabled!==!1&&(C.render(T,M,x,w),C.needsSwap!==!1&&(x=M,M=M===o?c:o))}if(m!==T.outputColorSpace||y!==T.toneMapping){m=T.outputColorSpace,y=T.toneMapping,h.defines={},oe.getTransfer(m)===ge&&(h.defines.SRGB_TRANSFER="");let P=P0[y];P&&(h.defines[P]=""),h.needsUpdate=!0}h.uniforms.tDiffuse.value=x.texture,T.setRenderTarget(v),T.render(d,f),v=null,g=!1},this.isCompositing=function(){return g},this.dispose=function(){a.dispose(),o!==null&&o.dispose(),c!==null&&c.dispose(),u.dispose(),h.dispose()}}var dd=new Qe,fd=new ei(1,1),pd=new pr(1),md=new ar,gd=new ka,xd=new fr,Xu=[],qu=[],Yu=new Float32Array(16),$u=new Float32Array(9),Zu=new Float32Array(4);function Os(i,t,e){let n=i[0];if(n<=0||n>0)return i;let s=t*e,r=Xu[s];if(r===void 0&&(r=new Float32Array(s),Xu[s]=r),t!==0){n.toArray(r,0);for(let a=1,o=0;a!==t;++a)o+=e,i[a].toArray(r,o)}return r}function Ue(i,t){if(i.length!==t.length)return!1;for(let e=0,n=i.length;e<n;e++)if(i[e]!==t[e])return!1;return!0}function Fe(i,t){for(let e=0,n=t.length;e<n;e++)i[e]=t[e]}function Ko(i,t){let e=qu[t];e===void 0&&(e=new Int32Array(t),qu[t]=e);for(let n=0;n!==t;++n)e[n]=i.allocateTextureUnit();return e}function L0(i,t){let e=this.cache;e[0]!==t&&(i.uniform1f(this.addr,t),e[0]=t)}function D0(i,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(i.uniform2f(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if(Ue(e,t))return;i.uniform2fv(this.addr,t),Fe(e,t)}}function N0(i,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(i.uniform3f(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else if(t.r!==void 0)(e[0]!==t.r||e[1]!==t.g||e[2]!==t.b)&&(i.uniform3f(this.addr,t.r,t.g,t.b),e[0]=t.r,e[1]=t.g,e[2]=t.b);else{if(Ue(e,t))return;i.uniform3fv(this.addr,t),Fe(e,t)}}function U0(i,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(i.uniform4f(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if(Ue(e,t))return;i.uniform4fv(this.addr,t),Fe(e,t)}}function F0(i,t){let e=this.cache,n=t.elements;if(n===void 0){if(Ue(e,t))return;i.uniformMatrix2fv(this.addr,!1,t),Fe(e,t)}else{if(Ue(e,n))return;Zu.set(n),i.uniformMatrix2fv(this.addr,!1,Zu),Fe(e,n)}}function O0(i,t){let e=this.cache,n=t.elements;if(n===void 0){if(Ue(e,t))return;i.uniformMatrix3fv(this.addr,!1,t),Fe(e,t)}else{if(Ue(e,n))return;$u.set(n),i.uniformMatrix3fv(this.addr,!1,$u),Fe(e,n)}}function B0(i,t){let e=this.cache,n=t.elements;if(n===void 0){if(Ue(e,t))return;i.uniformMatrix4fv(this.addr,!1,t),Fe(e,t)}else{if(Ue(e,n))return;Yu.set(n),i.uniformMatrix4fv(this.addr,!1,Yu),Fe(e,n)}}function k0(i,t){let e=this.cache;e[0]!==t&&(i.uniform1i(this.addr,t),e[0]=t)}function z0(i,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(i.uniform2i(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if(Ue(e,t))return;i.uniform2iv(this.addr,t),Fe(e,t)}}function G0(i,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(i.uniform3i(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else{if(Ue(e,t))return;i.uniform3iv(this.addr,t),Fe(e,t)}}function V0(i,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(i.uniform4i(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if(Ue(e,t))return;i.uniform4iv(this.addr,t),Fe(e,t)}}function H0(i,t){let e=this.cache;e[0]!==t&&(i.uniform1ui(this.addr,t),e[0]=t)}function W0(i,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y)&&(i.uniform2ui(this.addr,t.x,t.y),e[0]=t.x,e[1]=t.y);else{if(Ue(e,t))return;i.uniform2uiv(this.addr,t),Fe(e,t)}}function X0(i,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z)&&(i.uniform3ui(this.addr,t.x,t.y,t.z),e[0]=t.x,e[1]=t.y,e[2]=t.z);else{if(Ue(e,t))return;i.uniform3uiv(this.addr,t),Fe(e,t)}}function q0(i,t){let e=this.cache;if(t.x!==void 0)(e[0]!==t.x||e[1]!==t.y||e[2]!==t.z||e[3]!==t.w)&&(i.uniform4ui(this.addr,t.x,t.y,t.z,t.w),e[0]=t.x,e[1]=t.y,e[2]=t.z,e[3]=t.w);else{if(Ue(e,t))return;i.uniform4uiv(this.addr,t),Fe(e,t)}}function jo(i){return i.version===0&&(i.compareFunction=xc,i.needsUpdate=!0),i}function Y0(i,t,e){let n=this.cache,s=e.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s);let r;this.type===i.SAMPLER_2D_SHADOW?r=jo(fd):r=dd,e.setTexture2D(t||r,s)}function $0(i,t,e){let n=this.cache,s=e.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),e.setTexture3D(t||gd,s)}function Z0(i,t,e){let n=this.cache,s=e.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s);let r;this.type===i.SAMPLER_CUBE_SHADOW?r=jo(pd):r=xd,e.setTextureCube(t||r,s)}function J0(i,t,e){let n=this.cache,s=e.allocateTextureUnit();n[0]!==s&&(i.uniform1i(this.addr,s),n[0]=s),e.setTexture2DArray(t||md,s)}function K0(i){switch(i){case 5126:return L0;case 35664:return D0;case 35665:return N0;case 35666:return U0;case 35674:return F0;case 35675:return O0;case 35676:return B0;case 5124:case 35670:return k0;case 35667:case 35671:return z0;case 35668:case 35672:return G0;case 35669:case 35673:return V0;case 5125:return H0;case 36294:return W0;case 36295:return X0;case 36296:return q0;case 35678:case 36198:case 36298:case 36306:case 35682:return Y0;case 35679:case 36299:case 36307:return $0;case 35680:case 36300:case 36308:case 36293:return Z0;case 36289:case 36303:case 36311:case 36292:return J0}}function j0(i,t){i.uniform1fv(this.addr,t)}function Q0(i,t){let e=Os(t,this.size,2);i.uniform2fv(this.addr,e)}function tx(i,t){let e=Os(t,this.size,3);i.uniform3fv(this.addr,e)}function ex(i,t){let e=Os(t,this.size,4);i.uniform4fv(this.addr,e)}function nx(i,t){let e=Os(t,this.size,4);i.uniformMatrix2fv(this.addr,!1,e)}function ix(i,t){let e=Os(t,this.size,9);i.uniformMatrix3fv(this.addr,!1,e)}function sx(i,t){let e=Os(t,this.size,16);i.uniformMatrix4fv(this.addr,!1,e)}function rx(i,t){i.uniform1iv(this.addr,t)}function ax(i,t){i.uniform2iv(this.addr,t)}function ox(i,t){i.uniform3iv(this.addr,t)}function lx(i,t){i.uniform4iv(this.addr,t)}function cx(i,t){i.uniform1uiv(this.addr,t)}function hx(i,t){i.uniform2uiv(this.addr,t)}function ux(i,t){i.uniform3uiv(this.addr,t)}function dx(i,t){i.uniform4uiv(this.addr,t)}function fx(i,t,e){let n=this.cache,s=t.length,r=Ko(e,s);Ue(n,r)||(i.uniform1iv(this.addr,r),Fe(n,r));let a;this.type===i.SAMPLER_2D_SHADOW?a=jo(fd):a=dd;for(let o=0;o!==s;++o)e.setTexture2D(t[o]||a,r[o])}function px(i,t,e){let n=this.cache,s=t.length,r=Ko(e,s);Ue(n,r)||(i.uniform1iv(this.addr,r),Fe(n,r));for(let a=0;a!==s;++a)e.setTexture3D(t[a]||gd,r[a])}function mx(i,t,e){let n=this.cache,s=t.length,r=Ko(e,s);Ue(n,r)||(i.uniform1iv(this.addr,r),Fe(n,r));let a;this.type===i.SAMPLER_CUBE_SHADOW?a=jo(pd):a=xd;for(let o=0;o!==s;++o)e.setTextureCube(t[o]||a,r[o])}function gx(i,t,e){let n=this.cache,s=t.length,r=Ko(e,s);Ue(n,r)||(i.uniform1iv(this.addr,r),Fe(n,r));for(let a=0;a!==s;++a)e.setTexture2DArray(t[a]||md,r[a])}function xx(i){switch(i){case 5126:return j0;case 35664:return Q0;case 35665:return tx;case 35666:return ex;case 35674:return nx;case 35675:return ix;case 35676:return sx;case 5124:case 35670:return rx;case 35667:case 35671:return ax;case 35668:case 35672:return ox;case 35669:case 35673:return lx;case 5125:return cx;case 36294:return hx;case 36295:return ux;case 36296:return dx;case 35678:case 36198:case 36298:case 36306:case 35682:return fx;case 35679:case 36299:case 36307:return px;case 35680:case 36300:case 36308:case 36293:return mx;case 36289:case 36303:case 36311:case 36292:return gx}}var Pc=class{constructor(t,e,n){this.id=t,this.addr=n,this.cache=[],this.type=e.type,this.setValue=K0(e.type)}},Ic=class{constructor(t,e,n){this.id=t,this.addr=n,this.cache=[],this.type=e.type,this.size=e.size,this.setValue=xx(e.type)}},Lc=class{constructor(t){this.id=t,this.seq=[],this.map={}}setValue(t,e,n){let s=this.seq;for(let r=0,a=s.length;r!==a;++r){let o=s[r];o.setValue(t,e[o.id],n)}}},Cc=/(\w+)(\])?(\[|\.)?/g;function Ju(i,t){i.seq.push(t),i.map[t.id]=t}function _x(i,t,e){let n=i.name,s=n.length;for(Cc.lastIndex=0;;){let r=Cc.exec(n),a=Cc.lastIndex,o=r[1],c=r[2]==="]",l=r[3];if(c&&(o=o|0),l===void 0||l==="["&&a+2===s){Ju(e,l===void 0?new Pc(o,i,t):new Ic(o,i,t));break}else{let h=e.map[o];h===void 0&&(h=new Lc(o),Ju(e,h)),e=h}}}var Fs=class{constructor(t,e){this.seq=[],this.map={};let n=t.getProgramParameter(e,t.ACTIVE_UNIFORMS);for(let a=0;a<n;++a){let o=t.getActiveUniform(e,a),c=t.getUniformLocation(e,o.name);_x(o,c,this)}let s=[],r=[];for(let a of this.seq)a.type===t.SAMPLER_2D_SHADOW||a.type===t.SAMPLER_CUBE_SHADOW||a.type===t.SAMPLER_2D_ARRAY_SHADOW?s.push(a):r.push(a);s.length>0&&(this.seq=s.concat(r))}setValue(t,e,n,s){let r=this.map[e];r!==void 0&&r.setValue(t,n,s)}setOptional(t,e,n){let s=e[n];s!==void 0&&this.setValue(t,n,s)}static upload(t,e,n,s){for(let r=0,a=e.length;r!==a;++r){let o=e[r],c=n[o.id];c.needsUpdate!==!1&&o.setValue(t,c.value,s)}}static seqWithValue(t,e){let n=[];for(let s=0,r=t.length;s!==r;++s){let a=t[s];a.id in e&&n.push(a)}return n}};function Ku(i,t,e){let n=i.createShader(t);return i.shaderSource(n,e),i.compileShader(n),n}var yx=37297,vx=0;function bx(i,t){let e=i.split(`
`),n=[],s=Math.max(t-6,0),r=Math.min(t+6,e.length);for(let a=s;a<r;a++){let o=a+1;n.push(`${o===t?">":" "} ${o}: ${e[a]}`)}return n.join(`
`)}var ju=new Xt;function Mx(i){oe._getMatrix(ju,oe.workingColorSpace,i);let t=`mat3( ${ju.elements.map(e=>e.toFixed(4))} )`;switch(oe.getTransfer(i)){case sr:return[t,"LinearTransferOETF"];case ge:return[t,"sRGBTransferOETF"];default:return qt("WebGLProgram: Unsupported color space: ",i),[t,"LinearTransferOETF"]}}function Qu(i,t,e){let n=i.getShaderParameter(t,i.COMPILE_STATUS),r=(i.getShaderInfoLog(t)||"").trim();if(n&&r==="")return"";let a=/ERROR: 0:(\d+)/.exec(r);if(a){let o=parseInt(a[1]);return e.toUpperCase()+`

`+r+`

`+bx(i.getShaderSource(t),o)}else return r}function Sx(i,t){let e=Mx(t);return[`vec4 ${i}( vec4 value ) {`,`	return ${e[1]}( vec4( ${e[0]} * value.rgb, value.a ) );`,"}"].join(`
`)}var wx={[nc]:"Linear",[ic]:"Reinhard",[sc]:"Cineon",[Pn]:"ACESFilmic",[ac]:"AgX",[oc]:"Neutral",[rc]:"Custom"};function Tx(i,t){let e=wx[t];return e===void 0?(qt("WebGLProgram: Unsupported toneMapping:",t),"vec3 "+i+"( vec3 color ) { return LinearToneMapping( color ); }"):"vec3 "+i+"( vec3 color ) { return "+e+"ToneMapping( color ); }"}var Zo=new F;function Ax(){oe.getLuminanceCoefficients(Zo);let i=Zo.x.toFixed(4),t=Zo.y.toFixed(4),e=Zo.z.toFixed(4);return["float luminance( const in vec3 rgb ) {",`	const vec3 weights = vec3( ${i}, ${t}, ${e} );`,"	return dot( weights, rgb );","}"].join(`
`)}function Ex(i){return[i.extensionClipCullDistance?"#extension GL_ANGLE_clip_cull_distance : require":"",i.extensionMultiDraw?"#extension GL_ANGLE_multi_draw : require":""].filter(Vr).join(`
`)}function Cx(i){let t=[];for(let e in i){let n=i[e];n!==!1&&t.push("#define "+e+" "+n)}return t.join(`
`)}function Rx(i,t){let e={},n=i.getProgramParameter(t,i.ACTIVE_ATTRIBUTES);for(let s=0;s<n;s++){let r=i.getActiveAttrib(t,s),a=r.name,o=1;r.type===i.FLOAT_MAT2&&(o=2),r.type===i.FLOAT_MAT3&&(o=3),r.type===i.FLOAT_MAT4&&(o=4),e[a]={type:r.type,location:i.getAttribLocation(t,a),locationSize:o}}return e}function Vr(i){return i!==""}function td(i,t){let e=t.numSpotLightShadows+t.numSpotLightMaps-t.numSpotLightShadowsWithMaps;return i.replace(/NUM_SUN_LIGHTS/g,t.numSunLights).replace(/NUM_DIR_LIGHTS/g,t.numDirLights).replace(/NUM_SPOT_LIGHTS/g,t.numSpotLights).replace(/NUM_SPOT_LIGHT_MAPS/g,t.numSpotLightMaps).replace(/NUM_SPOT_LIGHT_COORDS/g,e).replace(/NUM_RECT_AREA_LIGHTS/g,t.numRectAreaLights).replace(/NUM_POINT_LIGHTS/g,t.numPointLights).replace(/NUM_HEMI_LIGHTS/g,t.numHemiLights).replace(/NUM_SUN_LIGHT_SHADOWS/g,t.numSunLightShadows).replace(/NUM_DIR_LIGHT_SHADOWS/g,t.numDirLightShadows).replace(/NUM_SPOT_LIGHT_SHADOWS_WITH_MAPS/g,t.numSpotLightShadowsWithMaps).replace(/NUM_SPOT_LIGHT_SHADOWS/g,t.numSpotLightShadows).replace(/NUM_POINT_LIGHT_SHADOWS/g,t.numPointLightShadows).replace(/NUM_LIGHT_PROBE_GRIDS/g,t.numLightProbeGrids)}function ed(i,t){return i.replace(/NUM_CLIPPING_PLANES/g,t.numClippingPlanes).replace(/UNION_CLIPPING_PLANES/g,t.numClippingPlanes-t.numClipIntersection)}var Px=/^[ \t]*#include +<([\w\d./]+)>/gm;function Dc(i){return i.replace(Px,Ix)}function Ix(i,t){let e=ie[t];if(e===void 0)throw new Error("THREE.WebGLProgram: Can not resolve #include <"+t+">");return Dc(e)}var Lx=/#pragma unroll_loop_start\s+for\s*\(\s*int\s+i\s*=\s*(\d+)\s*;\s*i\s*<\s*(\d+)\s*;\s*i\s*\+\+\s*\)\s*{([\s\S]+?)}\s+#pragma unroll_loop_end/g;function nd(i){return i.replace(Lx,Dx)}function Dx(i,t,e,n){let s="";for(let r=parseInt(t);r<parseInt(e);r++)s+=n.replace(/\[\s*i\s*\]/g,"[ "+r+" ]").replace(/UNROLLED_LOOP_INDEX/g,r);return s}function id(i){let t=`precision ${i.precision} float;
	precision ${i.precision} int;
	precision ${i.precision} sampler2D;
	precision ${i.precision} samplerCube;
	precision ${i.precision} sampler3D;
	precision ${i.precision} sampler2DArray;
	precision ${i.precision} sampler2DShadow;
	precision ${i.precision} samplerCubeShadow;
	precision ${i.precision} sampler2DArrayShadow;
	precision ${i.precision} isampler2D;
	precision ${i.precision} isampler3D;
	precision ${i.precision} isamplerCube;
	precision ${i.precision} isampler2DArray;
	precision ${i.precision} usampler2D;
	precision ${i.precision} usampler3D;
	precision ${i.precision} usamplerCube;
	precision ${i.precision} usampler2DArray;
	`;return i.precision==="highp"?t+=`
#define HIGH_PRECISION`:i.precision==="mediump"?t+=`
#define MEDIUM_PRECISION`:i.precision==="lowp"&&(t+=`
#define LOW_PRECISION`),t}var Nx={[Dr]:"SHADOWMAP_TYPE_PCF",[Ls]:"SHADOWMAP_TYPE_VSM"};function Ux(i){return Nx[i.shadowMapType]||"SHADOWMAP_TYPE_BASIC"}var Fx={[Ns]:"ENVMAP_MODE_REFRACTION"};function Ox(i){return i.envMap===!1?"ENVMAP_MODE_REFLECTION":Fx[i.envMapMode]||"ENVMAP_MODE_REFLECTION"}var Bx={[ec]:"ENVMAP_BLENDING_MULTIPLY",[mu]:"ENVMAP_BLENDING_MIX",[gu]:"ENVMAP_BLENDING_ADD"};function kx(i){return i.envMap===!1?"ENVMAP_BLENDING_NONE":Bx[i.combine]||"ENVMAP_BLENDING_NONE"}function zx(i,t,e,n){let s=i.getContext(),r=e.defines,a=e.vertexShader,o=e.fragmentShader,c=Ux(e),l=e.envMapPMREM===!0?"ENVMAP_TYPE_PMREM":"ENVMAP_TYPE_CUBE",u=Ox(e),h=kx(e),d=Ex(e),f=Cx(r),m=s.createProgram(),y,g,p=e.glslVersion?"#version "+e.glslVersion+`
`:"";e.isRawShaderMaterial?(y=["#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,f].filter(Vr).join(`
`),y.length>0&&(y+=`
`),g=["#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,f].filter(Vr).join(`
`),g.length>0&&(g+=`
`)):(y=[id(e),"#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,f,e.extensionClipCullDistance?"#define USE_CLIP_DISTANCE":"",e.batching?"#define USE_BATCHING":"",e.batchingColor?"#define USE_BATCHING_COLOR":"",e.instancing?"#define USE_INSTANCING":"",e.instancingColor?"#define USE_INSTANCING_COLOR":"",e.instancingMorph?"#define USE_INSTANCING_MORPH":"",e.useFog&&e.fog?"#define USE_FOG":"",e.useFog&&e.fogExp2?"#define FOG_EXP2":"",e.map?"#define USE_MAP":"",e.envMap?"#define USE_ENVMAP":"",e.envMap?"#define "+u:"",e.lightMap?"#define USE_LIGHTMAP":"",e.aoMap?"#define USE_AOMAP":"",e.bumpMap?"#define USE_BUMPMAP":"",e.normalMap?"#define USE_NORMALMAP":"",e.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",e.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",e.displacementMap?"#define USE_DISPLACEMENTMAP":"",e.emissiveMap?"#define USE_EMISSIVEMAP":"",e.anisotropy?"#define USE_ANISOTROPY":"",e.anisotropyMap?"#define USE_ANISOTROPYMAP":"",e.clearcoatMap?"#define USE_CLEARCOATMAP":"",e.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",e.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",e.diffuseRoughnessMap?"#define USE_DIFFUSE_ROUGHNESSMAP":"",e.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",e.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",e.specularMap?"#define USE_SPECULARMAP":"",e.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",e.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",e.roughnessMap?"#define USE_ROUGHNESSMAP":"",e.metalnessMap?"#define USE_METALNESSMAP":"",e.alphaMap?"#define USE_ALPHAMAP":"",e.alphaHash?"#define USE_ALPHAHASH":"",e.transmission?"#define USE_TRANSMISSION":"",e.transmissionMap?"#define USE_TRANSMISSIONMAP":"",e.thicknessMap?"#define USE_THICKNESSMAP":"",e.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",e.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",e.mapUv?"#define MAP_UV "+e.mapUv:"",e.alphaMapUv?"#define ALPHAMAP_UV "+e.alphaMapUv:"",e.lightMapUv?"#define LIGHTMAP_UV "+e.lightMapUv:"",e.aoMapUv?"#define AOMAP_UV "+e.aoMapUv:"",e.emissiveMapUv?"#define EMISSIVEMAP_UV "+e.emissiveMapUv:"",e.bumpMapUv?"#define BUMPMAP_UV "+e.bumpMapUv:"",e.normalMapUv?"#define NORMALMAP_UV "+e.normalMapUv:"",e.displacementMapUv?"#define DISPLACEMENTMAP_UV "+e.displacementMapUv:"",e.metalnessMapUv?"#define METALNESSMAP_UV "+e.metalnessMapUv:"",e.roughnessMapUv?"#define ROUGHNESSMAP_UV "+e.roughnessMapUv:"",e.anisotropyMapUv?"#define ANISOTROPYMAP_UV "+e.anisotropyMapUv:"",e.clearcoatMapUv?"#define CLEARCOATMAP_UV "+e.clearcoatMapUv:"",e.clearcoatNormalMapUv?"#define CLEARCOAT_NORMALMAP_UV "+e.clearcoatNormalMapUv:"",e.clearcoatRoughnessMapUv?"#define CLEARCOAT_ROUGHNESSMAP_UV "+e.clearcoatRoughnessMapUv:"",e.diffuseRoughnessMapUv?"#define DIFFUSE_ROUGHNESSMAP_UV "+e.diffuseRoughnessMapUv:"",e.iridescenceMapUv?"#define IRIDESCENCEMAP_UV "+e.iridescenceMapUv:"",e.iridescenceThicknessMapUv?"#define IRIDESCENCE_THICKNESSMAP_UV "+e.iridescenceThicknessMapUv:"",e.sheenColorMapUv?"#define SHEEN_COLORMAP_UV "+e.sheenColorMapUv:"",e.sheenRoughnessMapUv?"#define SHEEN_ROUGHNESSMAP_UV "+e.sheenRoughnessMapUv:"",e.specularMapUv?"#define SPECULARMAP_UV "+e.specularMapUv:"",e.specularColorMapUv?"#define SPECULAR_COLORMAP_UV "+e.specularColorMapUv:"",e.specularIntensityMapUv?"#define SPECULAR_INTENSITYMAP_UV "+e.specularIntensityMapUv:"",e.transmissionMapUv?"#define TRANSMISSIONMAP_UV "+e.transmissionMapUv:"",e.thicknessMapUv?"#define THICKNESSMAP_UV "+e.thicknessMapUv:"",e.vertexTangents&&e.flatShading===!1?"#define USE_TANGENT":"",e.vertexNormals?"#define HAS_NORMAL":"",e.vertexColors?"#define USE_COLOR":"",e.vertexAlphas?"#define USE_COLOR_ALPHA":"",e.vertexUv1s?"#define USE_UV1":"",e.vertexUv2s?"#define USE_UV2":"",e.vertexUv3s?"#define USE_UV3":"",e.pointsUvs?"#define USE_POINTS_UV":"",e.flatShading?"#define FLAT_SHADED":"",e.skinning?"#define USE_SKINNING":"",e.morphTargets?"#define USE_MORPHTARGETS":"",e.morphNormals&&e.flatShading===!1?"#define USE_MORPHNORMALS":"",e.morphColors?"#define USE_MORPHCOLORS":"",e.morphTargetsCount>0?"#define MORPHTARGETS_TEXTURE_STRIDE "+e.morphTextureStride:"",e.morphTargetsCount>0?"#define MORPHTARGETS_COUNT "+e.morphTargetsCount:"",e.doubleSided?"#define DOUBLE_SIDED":"",e.flipSided?"#define FLIP_SIDED":"",e.shadowMapEnabled?"#define USE_SHADOWMAP":"",e.shadowMapEnabled?"#define "+c:"",e.sizeAttenuation?"#define USE_SIZEATTENUATION":"",e.numLightProbes>0?"#define USE_LIGHT_PROBES":"",e.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",e.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 modelMatrix;","uniform mat4 modelViewMatrix;","uniform mat4 projectionMatrix;","uniform mat4 viewMatrix;","uniform mat3 normalMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;","#ifdef USE_INSTANCING","	attribute mat4 instanceMatrix;","#endif","#ifdef USE_INSTANCING_COLOR","	attribute vec3 instanceColor;","#endif","#ifdef USE_INSTANCING_MORPH","	uniform sampler2D morphTexture;","#endif","attribute vec3 position;","attribute vec3 normal;","attribute vec2 uv;","#ifdef USE_UV1","	attribute vec2 uv1;","#endif","#ifdef USE_UV2","	attribute vec2 uv2;","#endif","#ifdef USE_UV3","	attribute vec2 uv3;","#endif","#ifdef USE_TANGENT","	attribute vec4 tangent;","#endif","#if defined( USE_COLOR_ALPHA )","	attribute vec4 color;","#elif defined( USE_COLOR )","	attribute vec3 color;","#endif","#ifdef USE_SKINNING","	attribute vec4 skinIndex;","	attribute vec4 skinWeight;","#endif",`
`].filter(Vr).join(`
`),g=[id(e),"#define SHADER_TYPE "+e.shaderType,"#define SHADER_NAME "+e.shaderName,f,e.useFog&&e.fog?"#define USE_FOG":"",e.useFog&&e.fogExp2?"#define FOG_EXP2":"",e.alphaToCoverage?"#define ALPHA_TO_COVERAGE":"",e.map?"#define USE_MAP":"",e.matcap?"#define USE_MATCAP":"",e.envMap?"#define USE_ENVMAP":"",e.envMap?"#define "+l:"",e.envMap?"#define "+u:"",e.envMap?"#define "+h:"",e.envMapPMREM?"#define ENVMAP_MAX_LOD "+e.envMapMaxLod+".0":"",e.envMapPMREM?"#define ENVMAP_SIZE "+e.envMapSize+".0":"",e.lightMap?"#define USE_LIGHTMAP":"",e.aoMap?"#define USE_AOMAP":"",e.bumpMap?"#define USE_BUMPMAP":"",e.normalMap?"#define USE_NORMALMAP":"",e.normalMapObjectSpace?"#define USE_NORMALMAP_OBJECTSPACE":"",e.normalMapTangentSpace?"#define USE_NORMALMAP_TANGENTSPACE":"",e.packedNormalMap?"#define USE_PACKED_NORMALMAP":"",e.emissiveMap?"#define USE_EMISSIVEMAP":"",e.anisotropy?"#define USE_ANISOTROPY":"",e.anisotropyMap?"#define USE_ANISOTROPYMAP":"",e.clearcoat?"#define USE_CLEARCOAT":"",e.clearcoatMap?"#define USE_CLEARCOATMAP":"",e.clearcoatRoughnessMap?"#define USE_CLEARCOAT_ROUGHNESSMAP":"",e.clearcoatNormalMap?"#define USE_CLEARCOAT_NORMALMAP":"",e.diffuseRoughness?"#define USE_DIFFUSE_ROUGHNESS":"",e.diffuseRoughnessMap?"#define USE_DIFFUSE_ROUGHNESSMAP":"",e.dispersion?"#define USE_DISPERSION":"",e.retroreflection?"#define USE_RETROREFLECTION":"",e.iridescence?"#define USE_IRIDESCENCE":"",e.iridescenceMap?"#define USE_IRIDESCENCEMAP":"",e.iridescenceThicknessMap?"#define USE_IRIDESCENCE_THICKNESSMAP":"",e.specularMap?"#define USE_SPECULARMAP":"",e.specularColorMap?"#define USE_SPECULAR_COLORMAP":"",e.specularIntensityMap?"#define USE_SPECULAR_INTENSITYMAP":"",e.roughnessMap?"#define USE_ROUGHNESSMAP":"",e.metalnessMap?"#define USE_METALNESSMAP":"",e.alphaMap?"#define USE_ALPHAMAP":"",e.alphaTest?"#define USE_ALPHATEST":"",e.alphaHash?"#define USE_ALPHAHASH":"",e.sheen?"#define USE_SHEEN":"",e.sheenColorMap?"#define USE_SHEEN_COLORMAP":"",e.sheenRoughnessMap?"#define USE_SHEEN_ROUGHNESSMAP":"",e.transmission?"#define USE_TRANSMISSION":"",e.transmissionMap?"#define USE_TRANSMISSIONMAP":"",e.thicknessMap?"#define USE_THICKNESSMAP":"",e.vertexTangents&&e.flatShading===!1?"#define USE_TANGENT":"",e.vertexColors||e.instancingColor?"#define USE_COLOR":"",e.vertexAlphas||e.batchingColor?"#define USE_COLOR_ALPHA":"",e.vertexUv1s?"#define USE_UV1":"",e.vertexUv2s?"#define USE_UV2":"",e.vertexUv3s?"#define USE_UV3":"",e.pointsUvs?"#define USE_POINTS_UV":"",e.gradientMap?"#define USE_GRADIENTMAP":"",e.flatShading?"#define FLAT_SHADED":"",e.doubleSided?"#define DOUBLE_SIDED":"",e.flipSided?"#define FLIP_SIDED":"",e.shadowMapEnabled?"#define USE_SHADOWMAP":"",e.shadowMapEnabled?"#define "+c:"",e.premultipliedAlpha?"#define PREMULTIPLIED_ALPHA":"",e.numLightProbes>0?"#define USE_LIGHT_PROBES":"",e.decodeVideoTexture?"#define DECODE_VIDEO_TEXTURE":"",e.decodeVideoTextureEmissive?"#define DECODE_VIDEO_TEXTURE_EMISSIVE":"",e.logarithmicDepthBuffer?"#define USE_LOGARITHMIC_DEPTH_BUFFER":"",e.reversedDepthBuffer?"#define USE_REVERSED_DEPTH_BUFFER":"","uniform mat4 viewMatrix;","uniform vec3 cameraPosition;","uniform bool isOrthographic;",e.toneMapping!==Bn?"#define TONE_MAPPING":"",e.toneMapping!==Bn?ie.tonemapping_pars_fragment:"",e.toneMapping!==Bn?Tx("toneMapping",e.toneMapping):"",e.dithering?"#define DITHERING":"",e.opaque?"#define OPAQUE":"",ie.colorspace_pars_fragment,Sx("linearToOutputTexel",e.outputColorSpace),Ax(),e.useDepthPacking?"#define DEPTH_PACKING "+e.depthPacking:"",`
`].filter(Vr).join(`
`)),a=Dc(a),a=td(a,e),a=ed(a,e),o=Dc(o),o=td(o,e),o=ed(o,e),a=nd(a),o=nd(o),e.isRawShaderMaterial!==!0&&(p=`#version 300 es
`,y=[d,"#define attribute in","#define varying out","#define texture2D texture"].join(`
`)+`
`+y,g=["#define varying in",e.glslVersion===_c?"":"layout(location = 0) out highp vec4 pc_fragColor;",e.glslVersion===_c?"":"#define gl_FragColor pc_fragColor","#define gl_FragDepthEXT gl_FragDepth","#define texture2D texture","#define textureCube texture","#define texture2DProj textureProj","#define texture2DLodEXT textureLod","#define texture2DProjLodEXT textureProjLod","#define textureCubeLodEXT textureLod","#define texture2DGradEXT textureGrad","#define texture2DProjGradEXT textureProjGrad","#define textureCubeGradEXT textureGrad"].join(`
`)+`
`+g);let v=p+y+a,b=p+g+o,_=Ku(s,s.VERTEX_SHADER,v),T=Ku(s,s.FRAGMENT_SHADER,b);s.attachShader(m,_),s.attachShader(m,T),e.index0AttributeName!==void 0?s.bindAttribLocation(m,0,e.index0AttributeName):e.hasPositionAttribute===!0&&s.bindAttribLocation(m,0,"position"),s.linkProgram(m);function w(C){if(i.debug.checkShaderErrors){let k=s.getProgramInfoLog(m)||"",O=s.getShaderInfoLog(_)||"",j=s.getShaderInfoLog(T)||"",D=k.trim(),K=O.trim(),Q=j.trim(),Y=!0,J=!0;if(s.getProgramParameter(m,s.LINK_STATUS)===!1)if(Y=!1,typeof i.debug.onShaderError=="function")i.debug.onShaderError(s,m,_,T);else{let at=Qu(s,_,"vertex"),W=Qu(s,T,"fragment");Zt("WebGLProgram: Shader Error "+s.getError()+" - VALIDATE_STATUS "+s.getProgramParameter(m,s.VALIDATE_STATUS)+`

Material Name: `+C.name+`
Material Type: `+C.type+`

Program Info Log: `+D+`
`+at+`
`+W)}else D!==""?qt("WebGLProgram: Program Info Log:",D):(K===""||Q==="")&&(J=!1);J&&(C.diagnostics={runnable:Y,programLog:D,vertexShader:{log:K,prefix:y},fragmentShader:{log:Q,prefix:g}})}s.deleteShader(_),s.deleteShader(T),x=new Fs(s,m),M=Rx(s,m)}let x;this.getUniforms=function(){return x===void 0&&w(this),x};let M;this.getAttributes=function(){return M===void 0&&w(this),M};let P=e.rendererExtensionParallelShaderCompile===!1;return this.isReady=function(){return P===!1&&(P=s.getProgramParameter(m,yx)),P},this.destroy=function(){n.releaseStatesOfProgram(this),s.deleteProgram(m),this.program=void 0},this.type=e.shaderType,this.name=e.shaderName,this.id=vx++,this.cacheKey=t,this.usedTimes=1,this.program=m,this.vertexShader=_,this.fragmentShader=T,this}var Gx=0,Nc=class{constructor(){this.shaderCache=new Map,this.materialCache=new Map}update(t,e,n){let s=this._getShaderCacheForMaterial(t);return s.has(e)===!1&&(s.add(e),e.usedTimes++),s.has(n)===!1&&(s.add(n),n.usedTimes++),this}remove(t){let e=this.materialCache.get(t);for(let n of e)n.usedTimes--,n.usedTimes===0&&this.shaderCache.delete(n.code);return this.materialCache.delete(t),this}getVertexShaderStage(t){return this._getShaderStage(t.vertexShader)}getFragmentShaderStage(t){return this._getShaderStage(t.fragmentShader)}dispose(){this.shaderCache.clear(),this.materialCache.clear()}_getShaderCacheForMaterial(t){let e=this.materialCache,n=e.get(t);return n===void 0&&(n=new Set,e.set(t,n)),n}_getShaderStage(t){let e=this.shaderCache,n=e.get(t);return n===void 0&&(n=new Uc(t),e.set(t,n)),n}},Uc=class{constructor(t){this.id=Gx++,this.code=t,this.usedTimes=0}};function Vx(i){return i===Ti||i===kr||i===zr}function Hx(i,t,e,n,s,r){let a=new bs,o=new Nc,c=new Set,l=[],u=new Map,h=n.logarithmicDepthBuffer,d={MeshDepthMaterial:"depth",MeshDistanceMaterial:"distance",MeshNormalMaterial:"normal",MeshBasicMaterial:"basic",MeshLambertMaterial:"lambert",MeshPhongMaterial:"phong",MeshToonMaterial:"toon",MeshStandardMaterial:"physical",MeshPhysicalMaterial:"physical",MeshMatcapMaterial:"matcap",LineBasicMaterial:"basic",LineDashedMaterial:"dashed",PointsMaterial:"points",ShadowMaterial:"shadow",SpriteMaterial:"sprite"};function f(x){return c.add(x),x===0?"uv":`uv${x}`}function m(x,M,P,C,k,O){let j=C.fog,D=k.geometry,K=x.isMeshStandardMaterial||x.isMeshLambertMaterial||x.isMeshPhongMaterial?C.environment:null,Q=x.isMeshStandardMaterial||x.isMeshLambertMaterial&&!x.envMap||x.isMeshPhongMaterial&&!x.envMap,Y=t.get(x.envMap||K,Q),J=d[x.type],at=n.precision;x.precision!==null&&(at=n.getMaxPrecision(x.precision),at!==x.precision&&qt("WebGLProgram.getParameters:",x.precision,"not supported, using",at,"instead."));let W=D.morphAttributes.position||D.morphAttributes.normal||D.morphAttributes.color,tt=W!==void 0?W.length:0,vt=0;D.morphAttributes.position!==void 0&&(vt=1),D.morphAttributes.normal!==void 0&&(vt=2),D.morphAttributes.color!==void 0&&(vt=3);let wt,st,U,et;if(J){let Nt=zn[J];wt=Nt.vertexShader,st=Nt.fragmentShader}else{wt=x.vertexShader,st=x.fragmentShader;let Nt=o.getVertexShaderStage(x),_e=o.getFragmentShaderStage(x);o.update(x,Nt,_e),U=Nt.id,et=_e.id}let V=i.getRenderTarget(),N=i.state.buffers.depth.getReversed(),L=k.isInstancedMesh===!0,I=k.isBatchedMesh===!0,G=!!x.map,ht=!!x.matcap,ut=!!Y,E=ut&&Y.isPMREMTexture===!0,$=!!x.aoMap,B=!!x.lightMap,nt=!!x.bumpMap&&x.wireframe===!1,it=!!x.normalMap,pt=!!x.displacementMap,mt=!!x.emissiveMap,X=!!x.metalnessMap,Z=!!x.roughnessMap,bt=x.anisotropy>0,H=x.clearcoat>0,Qt=x.diffuseRoughness>0,jt=x.dispersion>0,se=x.retroreflectivity>0,Mt=x.iridescence>0,fe=x.sheen>0,Gt=x.transmission>0,$t=bt&&!!x.anisotropyMap,R=H&&!!x.clearcoatMap,S=H&&!!x.clearcoatNormalMap,rt=H&&!!x.clearcoatRoughnessMap,St=Qt&&!!x.diffuseRoughnessMap,yt=Mt&&!!x.iridescenceMap,Et=Mt&&!!x.iridescenceThicknessMap,Rt=fe&&!!x.sheenColorMap,gt=fe&&!!x.sheenRoughnessMap,xt=!!x.specularMap,Pt=!!x.specularColorMap,z=!!x.specularIntensityMap,_t=Gt&&!!x.transmissionMap,dt=Gt&&!!x.thicknessMap,It=!!x.gradientMap,At=!!x.alphaMap,Ct=x.alphaTest>0,Wt=!!x.alphaHash,ee=!!x.extensions,Tt=Bn;x.toneMapped&&(V===null||V.isXRRenderTarget===!0)&&(Tt=i.toneMapping);let Yt=E?Y.isCompressedCubeTexture?Y.image[0].mipmaps:Y.mipmaps:null,te={shaderID:J,shaderType:x.type,shaderName:x.name,vertexShader:wt,fragmentShader:st,defines:x.defines,customVertexShaderID:U,customFragmentShaderID:et,isRawShaderMaterial:x.isRawShaderMaterial===!0,glslVersion:x.glslVersion,precision:at,batching:I,batchingColor:I&&k._colorsTexture!==null,instancing:L,instancingColor:L&&k.instanceColor!==null,instancingMorph:L&&k.morphTexture!==null,outputColorSpace:V===null?i.outputColorSpace:V.isXRRenderTarget===!0?V.texture.colorSpace:oe.workingColorSpace,alphaToCoverage:!!x.alphaToCoverage,map:G,matcap:ht,envMap:ut,envMapMode:ut&&Y.mapping,envMapPMREM:E,envMapMaxLod:E?Yt.length-1:null,envMapSize:E?Yt[0].width:null,aoMap:$,lightMap:B,bumpMap:nt,normalMap:it,displacementMap:pt,emissiveMap:mt,normalMapObjectSpace:it&&x.normalMapType===yu,normalMapTangentSpace:it&&x.normalMapType===Yo,packedNormalMap:it&&x.normalMapType===Yo&&Vx(x.normalMap.format),metalnessMap:X,roughnessMap:Z,anisotropy:bt,anisotropyMap:$t,clearcoat:H,clearcoatMap:R,clearcoatNormalMap:S,clearcoatRoughnessMap:rt,diffuseRoughness:Qt,diffuseRoughnessMap:St,dispersion:jt,retroreflection:se,iridescence:Mt,iridescenceMap:yt,iridescenceThicknessMap:Et,sheen:fe,sheenColorMap:Rt,sheenRoughnessMap:gt,specularMap:xt,specularColorMap:Pt,specularIntensityMap:z,transmission:Gt,transmissionMap:_t,thicknessMap:dt,gradientMap:It,opaque:x.transparent===!1&&x.blending===Ds&&x.alphaToCoverage===!1,alphaMap:At,alphaTest:Ct,alphaHash:Wt,combine:x.combine,mapUv:G&&f(x.map.channel),aoMapUv:$&&f(x.aoMap.channel),lightMapUv:B&&f(x.lightMap.channel),bumpMapUv:nt&&f(x.bumpMap.channel),normalMapUv:it&&f(x.normalMap.channel),displacementMapUv:pt&&f(x.displacementMap.channel),emissiveMapUv:mt&&f(x.emissiveMap.channel),metalnessMapUv:X&&f(x.metalnessMap.channel),roughnessMapUv:Z&&f(x.roughnessMap.channel),anisotropyMapUv:$t&&f(x.anisotropyMap.channel),clearcoatMapUv:R&&f(x.clearcoatMap.channel),clearcoatNormalMapUv:S&&f(x.clearcoatNormalMap.channel),clearcoatRoughnessMapUv:rt&&f(x.clearcoatRoughnessMap.channel),diffuseRoughnessMapUv:St&&f(x.diffuseRoughnessMap.channel),iridescenceMapUv:yt&&f(x.iridescenceMap.channel),iridescenceThicknessMapUv:Et&&f(x.iridescenceThicknessMap.channel),sheenColorMapUv:Rt&&f(x.sheenColorMap.channel),sheenRoughnessMapUv:gt&&f(x.sheenRoughnessMap.channel),specularMapUv:xt&&f(x.specularMap.channel),specularColorMapUv:Pt&&f(x.specularColorMap.channel),specularIntensityMapUv:z&&f(x.specularIntensityMap.channel),transmissionMapUv:_t&&f(x.transmissionMap.channel),thicknessMapUv:dt&&f(x.thicknessMap.channel),alphaMapUv:At&&f(x.alphaMap.channel),vertexTangents:!!D.attributes.tangent&&(it||bt),vertexNormals:!!D.attributes.normal,vertexColors:x.vertexColors,vertexAlphas:x.vertexColors===!0&&!!D.attributes.color&&D.attributes.color.itemSize===4,pointsUvs:k.isPoints===!0&&!!D.attributes.uv&&(G||At),fog:!!j,useFog:x.fog===!0,fogExp2:!!j&&j.isFogExp2,flatShading:x.wireframe===!1&&(x.flatShading===!0||D.attributes.normal===void 0&&it===!1&&(x.isMeshLambertMaterial||x.isMeshPhongMaterial||x.isMeshStandardMaterial||x.isMeshPhysicalMaterial)),sizeAttenuation:x.sizeAttenuation===!0,logarithmicDepthBuffer:h,reversedDepthBuffer:N,skinning:k.isSkinnedMesh===!0,hasPositionAttribute:D.attributes.position!==void 0,morphTargets:D.morphAttributes.position!==void 0,morphNormals:D.morphAttributes.normal!==void 0,morphColors:D.morphAttributes.color!==void 0,morphTargetsCount:tt,morphTextureStride:vt,numSunLights:M.sun.length,numDirLights:M.directional.length,numPointLights:M.point.length,numSpotLights:M.spot.length,numSpotLightMaps:M.spotLightMap.length,numRectAreaLights:M.rectArea.length,numHemiLights:M.hemi.length,numSunLightShadows:M.sunShadowMap.length,numDirLightShadows:M.directionalShadowMap.length,numPointLightShadows:M.pointShadowMap.length,numSpotLightShadows:M.spotShadowMap.length,numSpotLightShadowsWithMaps:M.numSpotLightShadowsWithMaps,numLightProbes:M.numLightProbes,numLightProbeGrids:O.filter(Nt=>Nt.texture!==null).length,numClippingPlanes:r.numPlanes,numClipIntersection:r.numIntersection,dithering:x.dithering,shadowMapEnabled:i.shadowMap.enabled&&P.length>0,shadowMapType:i.shadowMap.type,toneMapping:Tt,decodeVideoTexture:G&&x.map.isVideoTexture===!0&&oe.getTransfer(x.map.colorSpace)===ge,decodeVideoTextureEmissive:mt&&x.emissiveMap.isVideoTexture===!0&&oe.getTransfer(x.emissiveMap.colorSpace)===ge,premultipliedAlpha:x.premultipliedAlpha,doubleSided:x.side===Ee,flipSided:x.side===We,useDepthPacking:x.depthPacking>=0,depthPacking:x.depthPacking||0,index0AttributeName:x.index0AttributeName,extensionClipCullDistance:ee&&x.extensions.clipCullDistance===!0&&e.has("WEBGL_clip_cull_distance"),extensionMultiDraw:(ee&&x.extensions.multiDraw===!0||I)&&e.has("WEBGL_multi_draw"),rendererExtensionParallelShaderCompile:e.has("KHR_parallel_shader_compile"),customProgramCacheKey:x.customProgramCacheKey()};return te.vertexUv1s=c.has(1),te.vertexUv2s=c.has(2),te.vertexUv3s=c.has(3),c.clear(),te}function y(x){let M=[];if(x.shaderID?M.push(x.shaderID):(M.push(x.customVertexShaderID),M.push(x.customFragmentShaderID)),x.defines!==void 0)for(let P in x.defines)M.push(P),M.push(x.defines[P]);return x.isRawShaderMaterial===!1&&(g(M,x),p(M,x),M.push(i.outputColorSpace)),M.push(x.customProgramCacheKey),M.join()}function g(x,M){x.push(M.precision),x.push(M.outputColorSpace),x.push(M.envMapMode),x.push(M.envMapPMREM),x.push(M.envMapMaxLod),x.push(M.envMapSize),x.push(M.mapUv),x.push(M.alphaMapUv),x.push(M.lightMapUv),x.push(M.aoMapUv),x.push(M.bumpMapUv),x.push(M.normalMapUv),x.push(M.displacementMapUv),x.push(M.emissiveMapUv),x.push(M.metalnessMapUv),x.push(M.roughnessMapUv),x.push(M.anisotropyMapUv),x.push(M.clearcoatMapUv),x.push(M.clearcoatNormalMapUv),x.push(M.clearcoatRoughnessMapUv),x.push(M.diffuseRoughnessMapUv),x.push(M.iridescenceMapUv),x.push(M.iridescenceThicknessMapUv),x.push(M.sheenColorMapUv),x.push(M.sheenRoughnessMapUv),x.push(M.specularMapUv),x.push(M.specularColorMapUv),x.push(M.specularIntensityMapUv),x.push(M.transmissionMapUv),x.push(M.thicknessMapUv),x.push(M.combine),x.push(M.fogExp2),x.push(M.sizeAttenuation),x.push(M.morphTargetsCount),x.push(M.numSunLights),x.push(M.numDirLights),x.push(M.numPointLights),x.push(M.numSpotLights),x.push(M.numSpotLightMaps),x.push(M.numHemiLights),x.push(M.numRectAreaLights),x.push(M.numSunLightShadows),x.push(M.numDirLightShadows),x.push(M.numPointLightShadows),x.push(M.numSpotLightShadows),x.push(M.numSpotLightShadowsWithMaps),x.push(M.numLightProbes),x.push(M.numLightProbeGrids),x.push(M.shadowMapType),x.push(M.toneMapping),x.push(M.numClippingPlanes),x.push(M.numClipIntersection),x.push(M.depthPacking)}function p(x,M){a.disableAll(),M.instancing&&a.enable(0),M.instancingColor&&a.enable(1),M.instancingMorph&&a.enable(2),M.matcap&&a.enable(3),M.envMap&&a.enable(4),M.normalMapObjectSpace&&a.enable(5),M.normalMapTangentSpace&&a.enable(6),M.clearcoat&&a.enable(7),M.iridescence&&a.enable(8),M.alphaTest&&a.enable(9),M.vertexColors&&a.enable(10),M.vertexAlphas&&a.enable(11),M.vertexUv1s&&a.enable(12),M.vertexUv2s&&a.enable(13),M.vertexUv3s&&a.enable(14),M.vertexTangents&&a.enable(15),M.anisotropy&&a.enable(16),M.alphaHash&&a.enable(17),M.batching&&a.enable(18),M.dispersion&&a.enable(19),M.retroreflection&&a.enable(24),M.diffuseRoughness&&a.enable(25),M.batchingColor&&a.enable(20),M.gradientMap&&a.enable(21),M.packedNormalMap&&a.enable(22),M.vertexNormals&&a.enable(23),x.push(a.mask),a.disableAll(),M.fog&&a.enable(0),M.useFog&&a.enable(1),M.flatShading&&a.enable(2),M.logarithmicDepthBuffer&&a.enable(3),M.reversedDepthBuffer&&a.enable(4),M.skinning&&a.enable(5),M.morphTargets&&a.enable(6),M.morphNormals&&a.enable(7),M.morphColors&&a.enable(8),M.premultipliedAlpha&&a.enable(9),M.shadowMapEnabled&&a.enable(10),M.doubleSided&&a.enable(11),M.flipSided&&a.enable(12),M.useDepthPacking&&a.enable(13),M.dithering&&a.enable(14),M.transmission&&a.enable(15),M.sheen&&a.enable(16),M.opaque&&a.enable(17),M.pointsUvs&&a.enable(18),M.decodeVideoTexture&&a.enable(19),M.decodeVideoTextureEmissive&&a.enable(20),M.alphaToCoverage&&a.enable(21),M.numLightProbeGrids>0&&a.enable(22),M.hasPositionAttribute&&a.enable(23),x.push(a.mask)}function v(x){let M=d[x.type],P;if(M){let C=zn[M];P=Bu.clone(C.uniforms)}else P=x.uniforms;return P}function b(x,M){let P=u.get(M);return P!==void 0?++P.usedTimes:(P=new zx(i,M,x,s),l.push(P),u.set(M,P)),P}function _(x){if(--x.usedTimes===0){let M=l.indexOf(x);l[M]=l[l.length-1],l.pop(),u.delete(x.cacheKey),x.destroy()}}function T(x){o.remove(x)}function w(){o.dispose()}return{getParameters:m,getProgramCacheKey:y,getUniforms:v,acquireProgram:b,releaseProgram:_,releaseShaderCache:T,programs:l,dispose:w}}function Wx(){let i=new WeakMap;function t(a){return i.has(a)}function e(a){let o=i.get(a);return o===void 0&&(o={},i.set(a,o)),o}function n(a){i.delete(a)}function s(a,o,c){i.get(a)[o]=c}function r(){i=new WeakMap}return{has:t,get:e,remove:n,update:s,dispose:r}}function Xx(i,t){return i.groupOrder!==t.groupOrder?i.groupOrder-t.groupOrder:i.renderOrder!==t.renderOrder?i.renderOrder-t.renderOrder:i.material.id!==t.material.id?i.material.id-t.material.id:i.materialVariant!==t.materialVariant?i.materialVariant-t.materialVariant:i.z!==t.z?i.z-t.z:i.id-t.id}function sd(i,t){return i.groupOrder!==t.groupOrder?i.groupOrder-t.groupOrder:i.renderOrder!==t.renderOrder?i.renderOrder-t.renderOrder:i.z!==t.z?t.z-i.z:i.id-t.id}function rd(){let i=[],t=0,e=[],n=[],s=[];function r(){t=0,e.length=0,n.length=0,s.length=0}function a(d){let f=0;return d.isInstancedMesh&&(f+=2),d.isSkinnedMesh&&(f+=1),f}function o(d,f,m,y,g,p){let v=i[t];return v===void 0?(v={id:d.id,object:d,geometry:f,material:m,materialVariant:a(d),groupOrder:y,renderOrder:d.renderOrder,z:g,group:p},i[t]=v):(v.id=d.id,v.object=d,v.geometry=f,v.material=m,v.materialVariant=a(d),v.groupOrder=y,v.renderOrder=d.renderOrder,v.z=g,v.group=p),t++,v}function c(d,f,m,y,g,p,v){v.reversedDepth===!0&&(g=-g);let b=o(d,f,m,y,g,p);m.transmission>0?n.push(b):m.transparent===!0?s.push(b):e.push(b)}function l(d,f,m,y,g,p){let v=o(d,f,m,y,g,p);m.transmission>0?n.unshift(v):m.transparent===!0?s.unshift(v):e.unshift(v)}function u(d,f){e.length>1&&e.sort(d||Xx),n.length>1&&n.sort(f||sd),s.length>1&&s.sort(f||sd)}function h(){for(let d=t,f=i.length;d<f;d++){let m=i[d];if(m.id===null)break;m.id=null,m.object=null,m.geometry=null,m.material=null,m.group=null}}return{opaque:e,transmissive:n,transparent:s,init:r,push:c,unshift:l,finish:h,sort:u}}function qx(){let i=new WeakMap;function t(n,s){let r=i.get(n),a;return r===void 0?(a=new rd,i.set(n,[a])):s>=r.length?(a=new rd,r.push(a)):a=r[s],a}function e(){i=new WeakMap}return{get:t,dispose:e}}function Yx(){let i={};return{get:function(t){if(i[t.id]!==void 0)return i[t.id];let e;switch(t.type){case"SunLight":case"DirectionalLight":e={direction:new F,color:new Vt};break;case"SpotLight":e={position:new F,direction:new F,color:new Vt,distance:0,coneCos:0,penumbraCos:0,decay:0};break;case"PointLight":e={position:new F,color:new Vt,distance:0,decay:0};break;case"HemisphereLight":e={direction:new F,skyColor:new Vt,groundColor:new Vt};break;case"RectAreaLight":e={color:new Vt,position:new F,halfWidth:new F,halfHeight:new F};break}return i[t.id]=e,e}}}function $x(){let i={};return{get:function(t){if(i[t.id]!==void 0)return i[t.id];let e;switch(t.type){case"SunLight":case"DirectionalLight":e={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new ft};break;case"SpotLight":e={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new ft};break;case"PointLight":e={shadowIntensity:1,shadowBias:0,shadowNormalBias:0,shadowRadius:1,shadowMapSize:new ft,shadowCameraNear:1,shadowCameraFar:1e3};break}return i[t.id]=e,e}}}var Zx=0;function Jx(i,t){return(t.castShadow?2:0)-(i.castShadow?2:0)+(t.map?1:0)-(i.map?1:0)}function Kx(i){let t=new Yx,e=$x(),n={version:0,hash:{sunLength:-1,directionalLength:-1,pointLength:-1,spotLength:-1,rectAreaLength:-1,hemiLength:-1,numSunShadows:-1,numDirectionalShadows:-1,numPointShadows:-1,numSpotShadows:-1,numSpotMaps:-1,numLightProbes:-1},ambient:[0,0,0],probe:[],sun:[],sunShadow:[],sunShadowMap:[],sunShadowMatrix:[],sunShadowCascade:[],directional:[],directionalShadow:[],directionalShadowMap:[],directionalShadowMatrix:[],spot:[],spotLightMap:[],spotShadow:[],spotShadowMap:[],spotLightMatrix:[],rectArea:[],rectAreaLTC1:null,rectAreaLTC2:null,point:[],pointShadow:[],pointShadowMap:[],pointShadowMatrix:[],hemi:[],numSpotLightShadowsWithMaps:0,numLightProbes:0};for(let l=0;l<9;l++)n.probe.push(new F);let s=new F,r=new xe,a=new xe;function o(l){let u=0,h=0,d=0;for(let O=0;O<9;O++)n.probe[O].set(0,0,0);let f=0,m=0,y=0,g=0,p=0,v=0,b=0,_=0,T=0,w=0,x=0,M=0,P=0,C=0;l.sort(Jx);for(let O=0,j=l.length;O<j;O++){let D=l[O],K=D.color,Q=D.intensity,Y=D.distance,J=null;if(D.shadow&&D.shadow.map&&(D.shadow.map.texture.format===Ti?J=D.shadow.map.texture:J=D.shadow.map.depthTexture||D.shadow.map.texture),D.isAmbientLight)u+=K.r*Q,h+=K.g*Q,d+=K.b*Q;else if(D.isLightProbe){for(let at=0;at<9;at++)n.probe[at].addScaledVector(D.sh.coefficients[at],Q);C++}else if(D.isSunLight){let at=t.get(D);if(at.color.copy(D.color).multiplyScalar(D.intensity),D.castShadow){let W=D.shadow,tt=e.get(D);tt.shadowIntensity=W.intensity,tt.shadowBias=W.bias,tt.shadowNormalBias=W.normalBias,tt.shadowRadius=W.radius,tt.shadowMapSize.copy(W.mapSize).multiply(W.getFrameExtents()),n.sunShadow[m]=tt,n.sunShadowMap[m]=J;let vt=W.getViewportCount();for(let wt=0;wt<vt;wt++)n.sunShadowMatrix[y+wt]=W.getMatrix(wt),n.sunShadowCascade[y+wt]=W._cascadeData[wt];y+=vt,m++}n.sun[f]=at,f++}else if(D.isDirectionalLight){let at=t.get(D);if(at.color.copy(D.color).multiplyScalar(D.intensity),D.castShadow){let W=D.shadow,tt=e.get(D);tt.shadowIntensity=W.intensity,tt.shadowBias=W.bias,tt.shadowNormalBias=W.normalBias,tt.shadowRadius=W.radius,tt.shadowMapSize=W.mapSize,n.directionalShadow[g]=tt,n.directionalShadowMap[g]=J,n.directionalShadowMatrix[g]=D.shadow.matrix,T++}n.directional[g]=at,g++}else if(D.isSpotLight){let at=t.get(D);at.position.setFromMatrixPosition(D.matrixWorld),at.color.copy(K).multiplyScalar(Q),at.distance=Y,at.coneCos=Math.cos(D.angle),at.penumbraCos=Math.cos(D.angle*(1-D.penumbra)),at.decay=D.decay,n.spot[v]=at;let W=D.shadow;if(D.map&&(n.spotLightMap[M]=D.map,M++,W.updateMatrices(D),D.castShadow&&P++),n.spotLightMatrix[v]=W.matrix,D.castShadow){let tt=e.get(D);tt.shadowIntensity=W.intensity,tt.shadowBias=W.bias,tt.shadowNormalBias=W.normalBias,tt.shadowRadius=W.radius,tt.shadowMapSize=W.mapSize,n.spotShadow[v]=tt,n.spotShadowMap[v]=J,x++}v++}else if(D.isRectAreaLight){let at=t.get(D);at.color.copy(K).multiplyScalar(Q),at.halfWidth.set(D.width*.5,0,0),at.halfHeight.set(0,D.height*.5,0),n.rectArea[b]=at,b++}else if(D.isPointLight){let at=t.get(D);if(at.color.copy(D.color).multiplyScalar(D.intensity),at.distance=D.distance,at.decay=D.decay,D.castShadow){let W=D.shadow,tt=e.get(D);tt.shadowIntensity=W.intensity,tt.shadowBias=W.bias,tt.shadowNormalBias=W.normalBias,tt.shadowRadius=W.radius,tt.shadowMapSize=W.mapSize,tt.shadowCameraNear=W.camera.near,tt.shadowCameraFar=W.camera.far,n.pointShadow[p]=tt,n.pointShadowMap[p]=J,n.pointShadowMatrix[p]=D.shadow.matrix,w++}n.point[p]=at,p++}else if(D.isHemisphereLight){let at=t.get(D);at.skyColor.copy(D.color).multiplyScalar(Q),at.groundColor.copy(D.groundColor).multiplyScalar(Q),n.hemi[_]=at,_++}}b>0&&(i.has("OES_texture_float_linear")===!0?(n.rectAreaLTC1=Ut.LTC_FLOAT_1,n.rectAreaLTC2=Ut.LTC_FLOAT_2):(n.rectAreaLTC1=Ut.LTC_HALF_1,n.rectAreaLTC2=Ut.LTC_HALF_2)),n.ambient[0]=u,n.ambient[1]=h,n.ambient[2]=d;let k=n.hash;(k.sunLength!==f||k.directionalLength!==g||k.pointLength!==p||k.spotLength!==v||k.rectAreaLength!==b||k.hemiLength!==_||k.numSunShadows!==m||k.numDirectionalShadows!==T||k.numPointShadows!==w||k.numSpotShadows!==x||k.numSpotMaps!==M||k.numLightProbes!==C)&&(n.sun.length=f,n.directional.length=g,n.spot.length=v,n.rectArea.length=b,n.point.length=p,n.hemi.length=_,n.sunShadow.length=m,n.sunShadowMap.length=m,n.sunShadowMatrix.length=y,n.sunShadowCascade.length=y,n.directionalShadow.length=T,n.directionalShadowMap.length=T,n.directionalShadowMatrix.length=T,n.pointShadow.length=w,n.pointShadowMap.length=w,n.pointShadowMatrix.length=w,n.spotShadow.length=x,n.spotShadowMap.length=x,n.spotLightMatrix.length=x+M-P,n.spotLightMap.length=M,n.numSpotLightShadowsWithMaps=P,n.numLightProbes=C,k.sunLength=f,k.directionalLength=g,k.pointLength=p,k.spotLength=v,k.rectAreaLength=b,k.hemiLength=_,k.numSunShadows=m,k.numDirectionalShadows=T,k.numPointShadows=w,k.numSpotShadows=x,k.numSpotMaps=M,k.numLightProbes=C,n.version=Zx++)}function c(l,u){let h=0,d=0,f=0,m=0,y=0,g=0,p=u.matrixWorldInverse;for(let v=0,b=l.length;v<b;v++){let _=l[v];if(_.isSunLight){let T=n.sun[h];T.direction.setFromMatrixPosition(_.matrixWorld),T.direction.transformDirection(p),h++}else if(_.isDirectionalLight){let T=n.directional[d];T.direction.setFromMatrixPosition(_.matrixWorld),s.setFromMatrixPosition(_.target.matrixWorld),T.direction.sub(s),T.direction.transformDirection(p),d++}else if(_.isSpotLight){let T=n.spot[m];T.position.setFromMatrixPosition(_.matrixWorld),T.position.applyMatrix4(p),T.direction.setFromMatrixPosition(_.matrixWorld),s.setFromMatrixPosition(_.target.matrixWorld),T.direction.sub(s),T.direction.transformDirection(p),m++}else if(_.isRectAreaLight){let T=n.rectArea[y];T.position.setFromMatrixPosition(_.matrixWorld),T.position.applyMatrix4(p),a.identity(),r.copy(_.matrixWorld),r.premultiply(p),a.extractRotation(r),T.halfWidth.set(_.width*.5,0,0),T.halfHeight.set(0,_.height*.5,0),T.halfWidth.applyMatrix4(a),T.halfHeight.applyMatrix4(a),y++}else if(_.isPointLight){let T=n.point[f];T.position.setFromMatrixPosition(_.matrixWorld),T.position.applyMatrix4(p),f++}else if(_.isHemisphereLight){let T=n.hemi[g];T.direction.setFromMatrixPosition(_.matrixWorld),T.direction.transformDirection(p),g++}}}return{setup:o,setupView:c,state:n}}function ad(i){let t=new Kx(i),e=[],n=[],s=[];function r(h){u.camera=h,e.length=0,n.length=0,s.length=0}function a(h){h.isLightProbeGrid?s.push(h):e.push(h)}function o(h){n.push(h)}function c(){t.setup(e),s.sort((h,d)=>h.priority-d.priority||h.id-d.id)}function l(h){t.setupView(e,h)}let u={lightsArray:e,shadowsArray:n,lightProbeGridArray:s,camera:null,lights:t,transmissionRenderTarget:{},textureUnits:0};return{init:r,state:u,setupLights:c,setupLightsView:l,pushLight:a,pushShadow:o}}function jx(i){let t=new WeakMap;function e(s,r=0){let a=t.get(s),o;return a===void 0?(o=new ad(i),t.set(s,[o])):r>=a.length?(o=new ad(i),a.push(o)):o=a[r],o}function n(){t=new WeakMap}return{get:e,dispose:n}}var Qx=`void main() {
	gl_Position = vec4( position, 1.0 );
}`,t_=`uniform sampler2D shadow_pass;
uniform vec2 resolution;
uniform float radius;
void main() {
	const float samples = float( VSM_SAMPLES );
	float mean = 0.0;
	float squared_mean = 0.0;
	float uvStride = samples <= 1.0 ? 0.0 : 2.0 / ( samples - 1.0 );
	float uvStart = samples <= 1.0 ? 0.0 : - 1.0;
	for ( float i = 0.0; i < samples; i ++ ) {
		float uvOffset = uvStart + i * uvStride;
		#ifdef HORIZONTAL_PASS
			vec2 distribution = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( uvOffset, 0.0 ) * radius ) / resolution ).rg;
			mean += distribution.x;
			squared_mean += distribution.y * distribution.y + distribution.x * distribution.x;
		#else
			float depth = texture2D( shadow_pass, ( gl_FragCoord.xy + vec2( 0.0, uvOffset ) * radius ) / resolution ).r;
			mean += depth;
			squared_mean += depth * depth;
		#endif
	}
	mean = mean / samples;
	squared_mean = squared_mean / samples;
	float std_dev = sqrt( max( 0.0, squared_mean - mean * mean ) );
	gl_FragColor = vec4( mean, std_dev, 0.0, 1.0 );
}`,e_=[new F(1,0,0),new F(-1,0,0),new F(0,1,0),new F(0,-1,0),new F(0,0,1),new F(0,0,-1)],n_=[new F(0,-1,0),new F(0,-1,0),new F(0,0,1),new F(0,0,-1),new F(0,-1,0),new F(0,-1,0)],od=new xe,Gr=new F,Rc=new F;function i_(i,t,e){let n=new _i,s=new ft,r=new ft,a=new Se,o=new Ya,c=new $a,l={},u=e.maxTextureSize,h={[wi]:We,[We]:wi,[Ee]:Ee},d=new dn({defines:{VSM_SAMPLES:8},uniforms:{shadow_pass:{value:null},resolution:{value:new ft},radius:{value:4}},vertexShader:Qx,fragmentShader:t_}),f=d.clone();f.defines.HORIZONTAL_PASS=1;let m=new de;m.setAttribute("position",new Mn(new Float32Array([-1,-1,.5,3,-1,.5,-1,3,.5]),3));let y=new he(m,d),g=this;this.enabled=!1,this.autoUpdate=!0,this.needsUpdate=!1,this.type=Dr;let p=this.type;this.render=function(w,x,M){if(g.enabled===!1||g.autoUpdate===!1&&g.needsUpdate===!1||w.length===0)return;this.type===Zl&&(qt("WebGLShadowMap: PCFSoftShadowMap has been removed. Using PCFShadowMap instead."),this.type=Dr);let P=i.getRenderTarget(),C=i.getActiveCubeFace(),k=i.getActiveMipmapLevel(),O=i.state;O.setBlending(Xi),O.buffers.depth.getReversed()===!0?O.buffers.color.setClear(0,0,0,0):O.buffers.color.setClear(1,1,1,1),O.buffers.depth.setTest(!0),O.setScissorTest(!1);let j=p!==this.type;j&&x.traverse(function(D){D.material&&(Array.isArray(D.material)?D.material.forEach(K=>K.needsUpdate=!0):D.material.needsUpdate=!0)});for(let D=0,K=w.length;D<K;D++){let Q=w[D],Y=Q.shadow;if(Y===void 0){qt("WebGLShadowMap:",Q,"has no shadow.");continue}if(Y.autoUpdate===!1&&Y.needsUpdate===!1)continue;s.copy(Y.mapSize);let J=Y.getFrameExtents();s.multiply(J),r.copy(Y.mapSize),(s.x>u||s.y>u)&&(s.x>u&&(r.x=Math.floor(u/J.x),s.x=r.x*J.x,Y.mapSize.x=r.x),s.y>u&&(r.y=Math.floor(u/J.y),s.y=r.y*J.y,Y.mapSize.y=r.y));let at=i.state.buffers.depth.getReversed();if(Y.camera._reversedDepth=at,Y.map===null||j===!0){if(Y.map!==null&&(Y.map.depthTexture!==null&&(Y.map.depthTexture.dispose(),Y.map.depthTexture=null),Y.map.dispose()),this.type===Ls){if(Q.isPointLight){qt("WebGLShadowMap: VSM shadow maps are not supported for PointLights. Use PCF or BasicShadowMap instead.");continue}Y.map=new cn(s.x,s.y,{format:Ti,type:Ln,minFilter:je,magFilter:je,generateMipmaps:!1}),Y.map.texture.name=Q.name+".shadowMap",Y.map.depthTexture=new ei(s.x,s.y,xn),Y.map.depthTexture.name=Q.name+".shadowMapDepth",Y.map.depthTexture.format=Kn,Y.map.depthTexture.compareFunction=null,Y.map.depthTexture.minFilter=Ge,Y.map.depthTexture.magFilter=Ge}else Q.isPointLight?(Y.map=new Hr(s.x),Y.map.depthTexture=new pr(s.x,In)):(Y.map=new cn(s.x,s.y),Y.map.depthTexture=new ei(s.x,s.y,In)),Y.map.depthTexture.name=Q.name+".shadowMap",Y.map.depthTexture.format=Kn,this.type===Dr?(Y.map.depthTexture.compareFunction=at?gc:mc,Y.map.depthTexture.minFilter=je,Y.map.depthTexture.magFilter=je):(Y.map.depthTexture.compareFunction=null,Y.map.depthTexture.minFilter=Ge,Y.map.depthTexture.magFilter=Ge);Y.camera.updateProjectionMatrix()}Y.map.isWebGLCubeRenderTarget!==!0&&(Y.map.width!==s.x||Y.map.height!==s.y)&&Y.map.setSize(s.x,s.y);let W=Y.map.isWebGLCubeRenderTarget?6:Y.getViewportCount();Q.isPointLight!==!0&&Y.updateMatrices(Q,M);let tt=(Y.camera.layers.mask&4294967294)!==0?Y.camera.layers:M.layers;for(let vt=0;vt<W;vt++){let wt=Y.getCamera(vt);if(Q.isPointLight){let st=Y.camera,U=Y.matrix,et=Q.distance||st.far;et!==st.far&&(st.far=et,st.updateProjectionMatrix()),Gr.setFromMatrixPosition(Q.matrixWorld),st.position.copy(Gr),Rc.copy(st.position),Rc.add(e_[vt]),st.up.copy(n_[vt]),st.lookAt(Rc),st.updateMatrixWorld(),U.makeTranslation(-Gr.x,-Gr.y,-Gr.z),od.multiplyMatrices(st.projectionMatrix,st.matrixWorldInverse),Y._frustum.setFromProjectionMatrix(od,st.coordinateSystem,st.reversedDepth)}if(Y.map.isWebGLCubeRenderTarget)i.setRenderTarget(Y.map,vt),i.clear();else{vt===0&&(i.setRenderTarget(Y.map),i.clear());let st=Y.getViewport(vt);a.set(r.x*st.x,r.y*st.y,r.x*st.z,r.y*st.w),O.viewport(a)}n=Y.getFrustum(vt),_(x,M,wt,Q,tt,this.type)}Y.isPointLightShadow!==!0&&this.type===Ls&&v(Y,M),Y.needsUpdate=!1}p=this.type,g.needsUpdate=!1,i.setRenderTarget(P,C,k)};function v(w,x){let M=t.update(y);d.defines.VSM_SAMPLES!==w.blurSamples&&(d.defines.VSM_SAMPLES=w.blurSamples,f.defines.VSM_SAMPLES=w.blurSamples,d.needsUpdate=!0,f.needsUpdate=!0),w.mapPass===null?w.mapPass=new cn(s.x,s.y,{format:Ti,type:Ln}):(w.mapPass.width!==w.map.width||w.mapPass.height!==w.map.height)&&w.mapPass.setSize(w.map.width,w.map.height),d.uniforms.shadow_pass.value=w.map.depthTexture,d.uniforms.resolution.value.set(w.map.width,w.map.height),d.uniforms.radius.value=w.radius,i.setRenderTarget(w.mapPass),i.clear(),i.renderBufferDirect(x,null,M,d,y,null),f.uniforms.shadow_pass.value=w.mapPass.texture,f.uniforms.resolution.value.set(w.map.width,w.map.height),f.uniforms.radius.value=w.radius,i.setRenderTarget(w.map),i.clear(),i.renderBufferDirect(x,null,M,f,y,null)}function b(w,x,M,P){let C=null,k=M.isPointLight===!0?w.customDistanceMaterial:w.customDepthMaterial;if(k!==void 0)C=k;else if(C=M.isPointLight===!0?c:o,i.localClippingEnabled&&x.clipShadows===!0&&Array.isArray(x.clippingPlanes)&&x.clippingPlanes.length!==0||x.displacementMap&&x.displacementScale!==0||x.alphaMap&&x.alphaTest>0||x.map&&x.alphaTest>0||x.alphaToCoverage===!0){let O=C.uuid,j=x.uuid,D=l[O];D===void 0&&(D={},l[O]=D);let K=D[j];K===void 0&&(K=C.clone(),D[j]=K,x.addEventListener("dispose",T)),C=K}if(C.visible=x.visible,C.wireframe=x.wireframe,P===Ls?C.side=x.shadowSide!==null?x.shadowSide:x.side:C.side=x.shadowSide!==null?x.shadowSide:h[x.side],C.alphaMap=x.alphaMap,C.alphaTest=x.alphaToCoverage===!0?.5:x.alphaTest,C.map=x.map,C.clipShadows=x.clipShadows,C.clippingPlanes=x.clippingPlanes,C.clipIntersection=x.clipIntersection,C.displacementMap=x.displacementMap,C.displacementScale=x.displacementScale,C.displacementBias=x.displacementBias,C.wireframeLinewidth=x.wireframeLinewidth,C.linewidth=x.linewidth,M.isPointLight===!0&&C.isMeshDistanceMaterial===!0){let O=i.properties.get(C);O.light=M}return C}function _(w,x,M,P,C,k){if(w.visible===!1)return;if(w.layers.test(C)&&(w.isMesh||w.isLine||w.isPoints)&&(w.castShadow||w.receiveShadow&&k===Ls)&&(!w.frustumCulled||w.intersectsFrustum(n))){w.modelViewMatrix.multiplyMatrices(M.matrixWorldInverse,w.matrixWorld);let D=t.update(w),K=w.material;if(Array.isArray(K)){let Q=D.groups;for(let Y=0,J=Q.length;Y<J;Y++){let at=Q[Y],W=K[at.materialIndex];if(W&&W.visible){let tt=b(w,W,P,k);w.onBeforeShadow(i,w,x,M,D,tt,at),i.renderBufferDirect(M,null,D,tt,w,at),w.onAfterShadow(i,w,x,M,D,tt,at)}}}else if(K.visible){let Q=b(w,K,P,k);w.onBeforeShadow(i,w,x,M,D,Q,null),i.renderBufferDirect(M,null,D,Q,w,null),w.onAfterShadow(i,w,x,M,D,Q,null)}}let j=w.children;for(let D=0,K=j.length;D<K;D++)_(j[D],x,M,P,C,k)}function T(w){w.target.removeEventListener("dispose",T);for(let M in l){let P=l[M],C=w.target.uuid;C in P&&(P[C].dispose(),delete P[C])}}}function s_(i,t){function e(){let z=!1,_t=new Se,dt=null,It=new Se(0,0,0,0);return{setMask:function(At){dt!==At&&!z&&(i.colorMask(At,At,At,At),dt=At)},setLocked:function(At){z=At},setClear:function(At,Ct,Wt,ee,Tt){Tt===!0&&(At*=ee,Ct*=ee,Wt*=ee),_t.set(At,Ct,Wt,ee),It.equals(_t)===!1&&(i.clearColor(At,Ct,Wt,ee),It.copy(_t))},reset:function(){z=!1,dt=null,It.set(-1,0,0,0)}}}function n(){let z=!1,_t=!1,dt=null,It=null,At=null;return{setReversed:function(Ct){if(_t!==Ct){let Wt=t.get("EXT_clip_control");Ct?Wt.clipControlEXT(Wt.LOWER_LEFT_EXT,Wt.ZERO_TO_ONE_EXT):Wt.clipControlEXT(Wt.LOWER_LEFT_EXT,Wt.NEGATIVE_ONE_TO_ONE_EXT),_t=Ct;let ee=At;At=null,this.setClear(ee)}},getReversed:function(){return _t},setTest:function(Ct){Ct?N(i.DEPTH_TEST):L(i.DEPTH_TEST)},setMask:function(Ct){dt!==Ct&&!z&&(i.depthMask(Ct),dt=Ct)},setFunc:function(Ct){if(_t&&(Ct=Pu[Ct]),It!==Ct){switch(Ct){case Ra:i.depthFunc(i.NEVER);break;case Pa:i.depthFunc(i.ALWAYS);break;case Ia:i.depthFunc(i.LESS);break;case ps:i.depthFunc(i.LEQUAL);break;case La:i.depthFunc(i.EQUAL);break;case Da:i.depthFunc(i.GEQUAL);break;case Na:i.depthFunc(i.GREATER);break;case Ua:i.depthFunc(i.NOTEQUAL);break;default:i.depthFunc(i.LEQUAL)}It=Ct}},setLocked:function(Ct){z=Ct},setClear:function(Ct){At!==Ct&&(At=Ct,_t&&(Ct=1-Ct),i.clearDepth(Ct))},reset:function(){z=!1,dt=null,It=null,At=null,_t=!1}}}function s(){let z=!1,_t=null,dt=null,It=null,At=null,Ct=null,Wt=null,ee=null,Tt=null;return{setTest:function(Yt){z||(Yt?N(i.STENCIL_TEST):L(i.STENCIL_TEST))},setMask:function(Yt){_t!==Yt&&!z&&(i.stencilMask(Yt),_t=Yt)},setFunc:function(Yt,te,Nt){(dt!==Yt||It!==te||At!==Nt)&&(i.stencilFunc(Yt,te,Nt),dt=Yt,It=te,At=Nt)},setOp:function(Yt,te,Nt){(Ct!==Yt||Wt!==te||ee!==Nt)&&(i.stencilOp(Yt,te,Nt),Ct=Yt,Wt=te,ee=Nt)},setLocked:function(Yt){z=Yt},setClear:function(Yt){Tt!==Yt&&(i.clearStencil(Yt),Tt=Yt)},reset:function(){z=!1,_t=null,dt=null,It=null,At=null,Ct=null,Wt=null,ee=null,Tt=null}}}let r=new e,a=new n,o=new s,c={},l=new WeakMap,u={},h={},d={},f=new WeakMap,m=[],y=null,g=!1,p=null,v=null,b=null,_=null,T=null,w=null,x=null,M=new Vt(0,0,0),P=0,C=!1,k=null,O=null,j=null,D=null,K=null,Q=i.getParameter(i.MAX_COMBINED_TEXTURE_IMAGE_UNITS),Y=!1,J=0,at=i.getParameter(i.VERSION);at.indexOf("WebGL")!==-1?(J=parseFloat(/^WebGL (\d)/.exec(at)[1]),Y=J>=1):at.indexOf("OpenGL ES")!==-1&&(J=parseFloat(/^OpenGL ES (\d)/.exec(at)[1]),Y=J>=2);let W=null,tt={},vt=i.getParameter(i.SCISSOR_BOX),wt=i.getParameter(i.VIEWPORT),st=new Se().fromArray(vt),U=new Se().fromArray(wt);function et(z,_t,dt,It){let At=new Uint8Array(4),Ct=i.createTexture();i.bindTexture(z,Ct),i.texParameteri(z,i.TEXTURE_MIN_FILTER,i.NEAREST),i.texParameteri(z,i.TEXTURE_MAG_FILTER,i.NEAREST);for(let Wt=0;Wt<dt;Wt++)z===i.TEXTURE_3D||z===i.TEXTURE_2D_ARRAY?i.texImage3D(_t,0,i.RGBA,1,1,It,0,i.RGBA,i.UNSIGNED_BYTE,At):i.texImage2D(_t+Wt,0,i.RGBA,1,1,0,i.RGBA,i.UNSIGNED_BYTE,At);return Ct}let V={};V[i.TEXTURE_2D]=et(i.TEXTURE_2D,i.TEXTURE_2D,1),V[i.TEXTURE_CUBE_MAP]=et(i.TEXTURE_CUBE_MAP,i.TEXTURE_CUBE_MAP_POSITIVE_X,6),V[i.TEXTURE_2D_ARRAY]=et(i.TEXTURE_2D_ARRAY,i.TEXTURE_2D_ARRAY,1,1),V[i.TEXTURE_3D]=et(i.TEXTURE_3D,i.TEXTURE_3D,1,1),r.setClear(0,0,0,1),a.setClear(1),o.setClear(0),N(i.DEPTH_TEST),a.setFunc(ps),nt(!1),it($l),N(i.CULL_FACE),$(Xi);function N(z){u[z]!==!0&&(i.enable(z),u[z]=!0)}function L(z){u[z]!==!1&&(i.disable(z),u[z]=!1)}function I(z,_t){return d[z]!==_t?(i.bindFramebuffer(z,_t),d[z]=_t,z===i.DRAW_FRAMEBUFFER&&(d[i.FRAMEBUFFER]=_t),z===i.FRAMEBUFFER&&(d[i.DRAW_FRAMEBUFFER]=_t),!0):!1}function G(z,_t){let dt=m,It=!1;if(z){dt=f.get(_t),dt===void 0&&(dt=[],f.set(_t,dt));let At=z.textures;if(dt.length!==At.length||dt[0]!==i.COLOR_ATTACHMENT0){for(let Ct=0,Wt=At.length;Ct<Wt;Ct++)dt[Ct]=i.COLOR_ATTACHMENT0+Ct;dt.length=At.length,It=!0}}else dt[0]!==i.BACK&&(dt[0]=i.BACK,It=!0);It&&i.drawBuffers(dt)}function ht(z){return y!==z?(i.useProgram(z),y=z,!0):!1}let ut={[qi]:i.FUNC_ADD,[jh]:i.FUNC_SUBTRACT,[Qh]:i.FUNC_REVERSE_SUBTRACT};ut[tu]=i.MIN,ut[eu]=i.MAX;let E={[nu]:i.ZERO,[iu]:i.ONE,[su]:i.SRC_COLOR,[Ql]:i.SRC_ALPHA,[hu]:i.SRC_ALPHA_SATURATE,[lu]:i.DST_COLOR,[au]:i.DST_ALPHA,[ru]:i.ONE_MINUS_SRC_COLOR,[tc]:i.ONE_MINUS_SRC_ALPHA,[cu]:i.ONE_MINUS_DST_COLOR,[ou]:i.ONE_MINUS_DST_ALPHA,[uu]:i.CONSTANT_COLOR,[du]:i.ONE_MINUS_CONSTANT_COLOR,[fu]:i.CONSTANT_ALPHA,[pu]:i.ONE_MINUS_CONSTANT_ALPHA};function $(z,_t,dt,It,At,Ct,Wt,ee,Tt,Yt){if(z===Xi){g===!0&&(L(i.BLEND),g=!1);return}if(g===!1&&(N(i.BLEND),g=!0),z!==Kh){if(z!==p||Yt!==C){if((v!==qi||T!==qi)&&(i.blendEquation(i.FUNC_ADD),v=qi,T=qi),Yt)switch(z){case Ds:i.blendFuncSeparate(i.ONE,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Jl:i.blendFunc(i.ONE,i.ONE);break;case Kl:i.blendFuncSeparate(i.ZERO,i.ONE_MINUS_SRC_COLOR,i.ZERO,i.ONE);break;case jl:i.blendFuncSeparate(i.DST_COLOR,i.ONE_MINUS_SRC_ALPHA,i.ZERO,i.ONE);break;default:Zt("WebGLState: Invalid blending: ",z);break}else switch(z){case Ds:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE_MINUS_SRC_ALPHA,i.ONE,i.ONE_MINUS_SRC_ALPHA);break;case Jl:i.blendFuncSeparate(i.SRC_ALPHA,i.ONE,i.ONE,i.ONE);break;case Kl:Zt("WebGLState: SubtractiveBlending requires material.premultipliedAlpha = true");break;case jl:Zt("WebGLState: MultiplyBlending requires material.premultipliedAlpha = true");break;default:Zt("WebGLState: Invalid blending: ",z);break}b=null,_=null,w=null,x=null,M.set(0,0,0),P=0,p=z,C=Yt}return}At=At||_t,Ct=Ct||dt,Wt=Wt||It,(_t!==v||At!==T)&&(i.blendEquationSeparate(ut[_t],ut[At]),v=_t,T=At),(dt!==b||It!==_||Ct!==w||Wt!==x)&&(i.blendFuncSeparate(E[dt],E[It],E[Ct],E[Wt]),b=dt,_=It,w=Ct,x=Wt),(ee.equals(M)===!1||Tt!==P)&&(i.blendColor(ee.r,ee.g,ee.b,Tt),M.copy(ee),P=Tt),p=z,C=!1}function B(z,_t){z.side===Ee?L(i.CULL_FACE):N(i.CULL_FACE);let dt=z.side===We;_t&&(dt=!dt),nt(dt),z.blending===Ds&&z.transparent===!1?$(Xi):$(z.blending,z.blendEquation,z.blendSrc,z.blendDst,z.blendEquationAlpha,z.blendSrcAlpha,z.blendDstAlpha,z.blendColor,z.blendAlpha,z.premultipliedAlpha),a.setFunc(z.depthFunc),a.setTest(z.depthTest),a.setMask(z.depthWrite),r.setMask(z.colorWrite);let It=z.stencilWrite;o.setTest(It),It&&(o.setMask(z.stencilWriteMask),o.setFunc(z.stencilFunc,z.stencilRef,z.stencilFuncMask),o.setOp(z.stencilFail,z.stencilZFail,z.stencilZPass)),mt(z.polygonOffset,z.polygonOffsetFactor,z.polygonOffsetUnits),z.alphaToCoverage===!0?N(i.SAMPLE_ALPHA_TO_COVERAGE):L(i.SAMPLE_ALPHA_TO_COVERAGE)}function nt(z){k!==z&&(z?i.frontFace(i.CW):i.frontFace(i.CCW),k=z)}function it(z){z!==Zh?(N(i.CULL_FACE),z!==O&&(z===$l?i.cullFace(i.BACK):z===Jh?i.cullFace(i.FRONT):i.cullFace(i.FRONT_AND_BACK))):L(i.CULL_FACE),O=z}function pt(z){z!==j&&(Y&&i.lineWidth(z),j=z)}function mt(z,_t,dt){z?(N(i.POLYGON_OFFSET_FILL),(D!==_t||K!==dt)&&(D=_t,K=dt,a.getReversed()&&(_t=-_t,dt=-dt),i.polygonOffset(_t,dt))):L(i.POLYGON_OFFSET_FILL)}function X(z){z?N(i.SCISSOR_TEST):L(i.SCISSOR_TEST)}function Z(z){z===void 0&&(z=i.TEXTURE0+Q-1),W!==z&&(i.activeTexture(z),W=z)}function bt(z,_t,dt){dt===void 0&&(W===null?dt=i.TEXTURE0+Q-1:dt=W);let It=tt[dt];It===void 0&&(It={type:void 0,texture:void 0},tt[dt]=It),(It.type!==z||It.texture!==_t)&&(W!==dt&&(i.activeTexture(dt),W=dt),i.bindTexture(z,_t||V[z]),It.type=z,It.texture=_t)}function H(){let z=tt[W];z!==void 0&&z.type!==void 0&&(i.bindTexture(z.type,null),z.type=void 0,z.texture=void 0)}function Qt(){try{i.compressedTexImage2D(...arguments)}catch(z){Zt("WebGLState:",z)}}function jt(){try{i.compressedTexImage3D(...arguments)}catch(z){Zt("WebGLState:",z)}}function se(){try{i.texSubImage2D(...arguments)}catch(z){Zt("WebGLState:",z)}}function Mt(){try{i.texSubImage3D(...arguments)}catch(z){Zt("WebGLState:",z)}}function fe(){try{i.compressedTexSubImage2D(...arguments)}catch(z){Zt("WebGLState:",z)}}function Gt(){try{i.compressedTexSubImage3D(...arguments)}catch(z){Zt("WebGLState:",z)}}function $t(){try{i.texStorage2D(...arguments)}catch(z){Zt("WebGLState:",z)}}function R(){try{i.texStorage3D(...arguments)}catch(z){Zt("WebGLState:",z)}}function S(){try{i.texImage2D(...arguments)}catch(z){Zt("WebGLState:",z)}}function rt(){try{i.texImage3D(...arguments)}catch(z){Zt("WebGLState:",z)}}function St(z){return h[z]!==void 0?h[z]:i.getParameter(z)}function yt(z,_t){h[z]!==_t&&(i.pixelStorei(z,_t),h[z]=_t)}function Et(z){st.equals(z)===!1&&(i.scissor(z.x,z.y,z.z,z.w),st.copy(z))}function Rt(z){U.equals(z)===!1&&(i.viewport(z.x,z.y,z.z,z.w),U.copy(z))}function gt(z,_t){let dt=l.get(_t);dt===void 0&&(dt=new WeakMap,l.set(_t,dt));let It=dt.get(z);It===void 0&&(It=i.getUniformBlockIndex(_t,z.name),i.uniformBlockBinding(_t,It,It),dt.set(z,It))}function xt(z,_t,dt){let At=l.get(_t).get(z);c[At]!==dt&&(i.bindBufferBase(i.UNIFORM_BUFFER,At,dt),c[At]=dt)}function Pt(){i.disable(i.BLEND),i.disable(i.CULL_FACE),i.disable(i.DEPTH_TEST),i.disable(i.POLYGON_OFFSET_FILL),i.disable(i.SCISSOR_TEST),i.disable(i.STENCIL_TEST),i.disable(i.SAMPLE_ALPHA_TO_COVERAGE),i.blendEquation(i.FUNC_ADD),i.blendFunc(i.ONE,i.ZERO),i.blendFuncSeparate(i.ONE,i.ZERO,i.ONE,i.ZERO),i.blendColor(0,0,0,0),i.colorMask(!0,!0,!0,!0),i.clearColor(0,0,0,0),i.depthMask(!0),i.depthFunc(i.LESS),a.setReversed(!1),i.clearDepth(1),i.stencilMask(4294967295),i.stencilFunc(i.ALWAYS,0,4294967295),i.stencilOp(i.KEEP,i.KEEP,i.KEEP),i.clearStencil(0),i.cullFace(i.BACK),i.frontFace(i.CCW),i.polygonOffset(0,0),i.activeTexture(i.TEXTURE0),i.bindFramebuffer(i.FRAMEBUFFER,null),i.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),i.bindFramebuffer(i.READ_FRAMEBUFFER,null),i.useProgram(null),i.lineWidth(1),i.scissor(0,0,i.canvas.width,i.canvas.height),i.viewport(0,0,i.canvas.width,i.canvas.height),i.pixelStorei(i.PACK_ALIGNMENT,4),i.pixelStorei(i.UNPACK_ALIGNMENT,4),i.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,!1),i.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,!1),i.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,i.BROWSER_DEFAULT_WEBGL),i.pixelStorei(i.PACK_ROW_LENGTH,0),i.pixelStorei(i.PACK_SKIP_PIXELS,0),i.pixelStorei(i.PACK_SKIP_ROWS,0),i.pixelStorei(i.UNPACK_ROW_LENGTH,0),i.pixelStorei(i.UNPACK_IMAGE_HEIGHT,0),i.pixelStorei(i.UNPACK_SKIP_PIXELS,0),i.pixelStorei(i.UNPACK_SKIP_ROWS,0),i.pixelStorei(i.UNPACK_SKIP_IMAGES,0),u={},h={},W=null,tt={},d={},c={},f=new WeakMap,m=[],y=null,g=!1,p=null,v=null,b=null,_=null,T=null,w=null,x=null,M=new Vt(0,0,0),P=0,C=!1,k=null,O=null,j=null,D=null,K=null,st.set(0,0,i.canvas.width,i.canvas.height),U.set(0,0,i.canvas.width,i.canvas.height),r.reset(),a.reset(),o.reset()}return{buffers:{color:r,depth:a,stencil:o},enable:N,disable:L,bindFramebuffer:I,drawBuffers:G,useProgram:ht,setBlending:$,setMaterial:B,setFlipSided:nt,setCullFace:it,setLineWidth:pt,setPolygonOffset:mt,setScissorTest:X,activeTexture:Z,bindTexture:bt,unbindTexture:H,compressedTexImage2D:Qt,compressedTexImage3D:jt,texImage2D:S,texImage3D:rt,pixelStorei:yt,getParameter:St,updateUBOMapping:gt,uniformBlockBinding:xt,texStorage2D:$t,texStorage3D:R,texSubImage2D:se,texSubImage3D:Mt,compressedTexSubImage2D:fe,compressedTexSubImage3D:Gt,scissor:Et,viewport:Rt,reset:Pt}}function r_(i,t,e,n,s,r,a){let o=t.has("WEBGL_multisampled_render_to_texture")?t.get("WEBGL_multisampled_render_to_texture"):null,c=typeof navigator>"u"?!1:/OculusBrowser/g.test(navigator.userAgent),l=new ft,u=new WeakMap,h=new Set,d,f=new WeakMap,m=new Set,y=new FinalizationRegistry(R=>m.delete(R)),g=!1;try{g=typeof OffscreenCanvas<"u"&&new OffscreenCanvas(1,1).getContext("2d")!==null}catch{}function p(R,S){return g?new OffscreenCanvas(R,S):rr("canvas")}function v(R,S,rt){let St=1,yt=$t(R);if((yt.width>rt||yt.height>rt)&&(St=rt/Math.max(yt.width,yt.height)),St<1)if(typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement||typeof HTMLCanvasElement<"u"&&R instanceof HTMLCanvasElement||typeof ImageBitmap<"u"&&R instanceof ImageBitmap||typeof VideoFrame<"u"&&R instanceof VideoFrame){let Et=Math.floor(St*yt.width),Rt=Math.floor(St*yt.height);d===void 0&&(d=p(Et,Rt));let gt=S?p(Et,Rt):d;return gt.width=Et,gt.height=Rt,gt.getContext("2d").drawImage(R,0,0,Et,Rt),qt("WebGLRenderer: Texture has been resized from ("+yt.width+"x"+yt.height+") to ("+Et+"x"+Rt+")."),gt}else return"data"in R&&qt("WebGLRenderer: Image in DataTexture is too big ("+yt.width+"x"+yt.height+")."),R;return R}function b(R){return R.generateMipmaps}function _(R){i.generateMipmap(R)}function T(R){return R.isWebGLCubeRenderTarget?i.TEXTURE_CUBE_MAP:R.isWebGL3DRenderTarget?i.TEXTURE_3D:R.isWebGLArrayRenderTarget||R.isCompressedArrayTexture?i.TEXTURE_2D_ARRAY:i.TEXTURE_2D}function w(R,S,rt,St,yt,Et=!1){if(R!==null){if(i[R]!==void 0)return i[R];qt("WebGLRenderer: Attempt to use non-existing WebGL internal format '"+R+"'")}let Rt;St&&(Rt=t.get("EXT_texture_norm16"),Rt||qt("WebGLRenderer: Unable to use normalized textures without EXT_texture_norm16 extension"));let gt=S;if(S===i.RED&&(rt===i.FLOAT&&(gt=i.R32F),rt===i.HALF_FLOAT&&(gt=i.R16F),rt===i.UNSIGNED_BYTE&&(gt=i.R8),rt===i.UNSIGNED_SHORT&&Rt&&(gt=Rt.R16_EXT),rt===i.SHORT&&Rt&&(gt=Rt.R16_SNORM_EXT)),S===i.RED_INTEGER&&(rt===i.UNSIGNED_BYTE&&(gt=i.R8UI),rt===i.UNSIGNED_SHORT&&(gt=i.R16UI),rt===i.UNSIGNED_INT&&(gt=i.R32UI),rt===i.BYTE&&(gt=i.R8I),rt===i.SHORT&&(gt=i.R16I),rt===i.INT&&(gt=i.R32I)),S===i.RG&&(rt===i.FLOAT&&(gt=i.RG32F),rt===i.HALF_FLOAT&&(gt=i.RG16F),rt===i.UNSIGNED_BYTE&&(gt=i.RG8),rt===i.UNSIGNED_SHORT&&Rt&&(gt=Rt.RG16_EXT),rt===i.SHORT&&Rt&&(gt=Rt.RG16_SNORM_EXT)),S===i.RG_INTEGER&&(rt===i.UNSIGNED_BYTE&&(gt=i.RG8UI),rt===i.UNSIGNED_SHORT&&(gt=i.RG16UI),rt===i.UNSIGNED_INT&&(gt=i.RG32UI),rt===i.BYTE&&(gt=i.RG8I),rt===i.SHORT&&(gt=i.RG16I),rt===i.INT&&(gt=i.RG32I)),S===i.RGB_INTEGER&&(rt===i.UNSIGNED_BYTE&&(gt=i.RGB8UI),rt===i.UNSIGNED_SHORT&&(gt=i.RGB16UI),rt===i.UNSIGNED_INT&&(gt=i.RGB32UI),rt===i.BYTE&&(gt=i.RGB8I),rt===i.SHORT&&(gt=i.RGB16I),rt===i.INT&&(gt=i.RGB32I)),S===i.RGBA_INTEGER&&(rt===i.UNSIGNED_BYTE&&(gt=i.RGBA8UI),rt===i.UNSIGNED_SHORT&&(gt=i.RGBA16UI),rt===i.UNSIGNED_INT&&(gt=i.RGBA32UI),rt===i.BYTE&&(gt=i.RGBA8I),rt===i.SHORT&&(gt=i.RGBA16I),rt===i.INT&&(gt=i.RGBA32I)),S===i.RGB&&(rt===i.UNSIGNED_SHORT&&Rt&&(gt=Rt.RGB16_EXT),rt===i.SHORT&&Rt&&(gt=Rt.RGB16_SNORM_EXT),rt===i.UNSIGNED_INT_5_9_9_9_REV&&(gt=i.RGB9_E5),rt===i.UNSIGNED_INT_10F_11F_11F_REV&&(gt=i.R11F_G11F_B10F)),S===i.RGBA){let xt=Et?sr:oe.getTransfer(yt);rt===i.FLOAT&&(gt=i.RGBA32F),rt===i.HALF_FLOAT&&(gt=i.RGBA16F),rt===i.UNSIGNED_BYTE&&(gt=xt===ge?i.SRGB8_ALPHA8:i.RGBA8),rt===i.UNSIGNED_SHORT&&Rt&&(gt=Rt.RGBA16_EXT),rt===i.SHORT&&Rt&&(gt=Rt.RGBA16_SNORM_EXT),rt===i.UNSIGNED_SHORT_4_4_4_4&&(gt=i.RGBA4),rt===i.UNSIGNED_SHORT_5_5_5_1&&(gt=i.RGB5_A1)}return(gt===i.R16F||gt===i.R32F||gt===i.RG16F||gt===i.RG32F||gt===i.RGBA16F||gt===i.RGBA32F)&&t.get("EXT_color_buffer_float"),gt}function x(R,S){let rt;return R?S===null||S===In||S===$i?rt=i.DEPTH24_STENCIL8:S===xn?rt=i.DEPTH32F_STENCIL8:S===Us&&(rt=i.DEPTH24_STENCIL8,qt("DepthTexture: 16 bit depth attachment is not supported with stencil. Using 24-bit attachment.")):S===null||S===In||S===$i?rt=i.DEPTH_COMPONENT24:S===xn?rt=i.DEPTH_COMPONENT32F:S===Us&&(rt=i.DEPTH_COMPONENT16),rt}function M(R,S){return b(R)===!0||R.isFramebufferTexture&&R.minFilter!==Ge&&R.minFilter!==je?Math.log2(Math.max(S.width,S.height))+1:R.mipmaps.length>0?R.mipmaps.length:R.isCompressedTexture&&Array.isArray(R.image)?S.mipmaps.length:1}function P(R){let S=new WeakRef(R);n.get(R).__ref=S,m.add(S),y.register(R,S,S)}function C(R){let S=n.get(R).__ref;S!==void 0&&(m.delete(S),y.unregister(S))}function k(R){j(R.target)}function O(R){D(R.target)}function j(R){R.removeEventListener("dispose",k),C(R),Q(R),R.isVideoTexture&&u.delete(R),R.isHTMLTexture&&h.delete(R)}function D(R){R.removeEventListener("dispose",O),C(R),J(R)}function K(){for(let R of m){let S=R.deref();S!==void 0&&(S.isRenderTarget===!0?D(S):j(S))}m.clear(),h.clear()}function Q(R){let S=n.get(R);if(S.__webglInit===void 0)return;let rt=R.source,St=f.get(rt);if(St){let yt=St[S.__cacheKey];yt.usedTimes--,yt.usedTimes===0&&Y(R),Object.keys(St).length===0&&f.delete(rt)}n.remove(R)}function Y(R){let S=n.get(R);i.deleteTexture(S.__webglTexture);let rt=R.source,St=f.get(rt);delete St[S.__cacheKey],a.memory.textures--}function J(R){let S=n.get(R);S.__depthDisposeCallback&&S.__depthDisposeCallback();let rt=R.depthTexture;if(rt&&rt.renderTarget===R&&j(rt),R.isWebGLCubeRenderTarget)for(let yt=0;yt<6;yt++){if(Array.isArray(S.__webglFramebuffer[yt]))for(let Et=0;Et<S.__webglFramebuffer[yt].length;Et++)i.deleteFramebuffer(S.__webglFramebuffer[yt][Et]);else i.deleteFramebuffer(S.__webglFramebuffer[yt]);S.__webglDepthbuffer&&i.deleteRenderbuffer(S.__webglDepthbuffer[yt])}else{if(Array.isArray(S.__webglFramebuffer))for(let yt=0;yt<S.__webglFramebuffer.length;yt++)i.deleteFramebuffer(S.__webglFramebuffer[yt]);else i.deleteFramebuffer(S.__webglFramebuffer);if(S.__webglDepthbuffer&&i.deleteRenderbuffer(S.__webglDepthbuffer),S.__webglMultisampledFramebuffer&&i.deleteFramebuffer(S.__webglMultisampledFramebuffer),S.__webglColorRenderbuffer)for(let yt=0;yt<S.__webglColorRenderbuffer.length;yt++)S.__webglColorRenderbuffer[yt]&&i.deleteRenderbuffer(S.__webglColorRenderbuffer[yt]);S.__webglDepthRenderbuffer&&i.deleteRenderbuffer(S.__webglDepthRenderbuffer)}let St=R.textures;for(let yt=0,Et=St.length;yt<Et;yt++){let Rt=n.get(St[yt]);Rt.__webglTexture&&(i.deleteTexture(Rt.__webglTexture),a.memory.textures--),n.remove(St[yt])}n.remove(R)}let at=0;function W(){at=0}function tt(){return at}function vt(R){at=R}function wt(){let R=at;return R>=s.maxTextures&&qt("WebGLTextures: Trying to use "+(R+1)+" texture units while this GPU supports only "+s.maxTextures),at+=1,R}function st(R){let S=[];return S.push(R.wrapS),S.push(R.wrapT),S.push(R.wrapR||0),S.push(R.magFilter),S.push(R.minFilter),S.push(R.anisotropy),S.push(R.internalFormat),S.push(R.format),S.push(R.type),S.push(R.generateMipmaps),S.push(R.mipmapsAutoUpdate),S.push(R.premultiplyAlpha),S.push(R.flipY),S.push(R.unpackAlignment),S.push(R.colorSpace),S.join()}function U(R,S){let rt=n.get(R);if(R.isVideoTexture&&fe(R),R.isRenderTargetTexture===!1&&R.isExternalTexture!==!0&&R.version>0&&rt.__version!==R.version){let St=R.image;if(St===null)qt("WebGLRenderer: Texture marked for update but no image data found.");else if(St.complete===!1)qt("WebGLRenderer: Texture marked for update but image is incomplete");else{B(rt,R,S);return}}else R.isExternalTexture&&(rt.__webglTexture=R.sourceTexture?R.sourceTexture:null);e.bindTexture(i.TEXTURE_2D,rt.__webglTexture,i.TEXTURE0+S)}function et(R,S){let rt=n.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&rt.__version!==R.version){B(rt,R,S);return}else R.isExternalTexture&&(rt.__webglTexture=R.sourceTexture?R.sourceTexture:null);e.bindTexture(i.TEXTURE_2D_ARRAY,rt.__webglTexture,i.TEXTURE0+S)}function V(R,S){let rt=n.get(R);if(R.isRenderTargetTexture===!1&&R.version>0&&rt.__version!==R.version){B(rt,R,S);return}e.bindTexture(i.TEXTURE_3D,rt.__webglTexture,i.TEXTURE0+S)}function N(R,S){let rt=n.get(R);if(R.version>0&&rt.__version!==R.version){nt(rt,R,S);return}e.bindTexture(i.TEXTURE_CUBE_MAP,rt.__webglTexture,i.TEXTURE0+S)}let L={[ms]:i.REPEAT,[mn]:i.CLAMP_TO_EDGE,[gs]:i.MIRRORED_REPEAT},I={[Ge]:i.NEAREST,[xu]:i.NEAREST_MIPMAP_NEAREST,[Nr]:i.NEAREST_MIPMAP_LINEAR,[je]:i.LINEAR,[lo]:i.LINEAR_MIPMAP_NEAREST,[si]:i.LINEAR_MIPMAP_LINEAR},G={[bu]:i.NEVER,[xc]:i.ALWAYS,[Mu]:i.LESS,[mc]:i.LEQUAL,[Su]:i.EQUAL,[gc]:i.GEQUAL,[wu]:i.GREATER,[Tu]:i.NOTEQUAL};function ht(R,S){if(S.type===xn&&t.has("OES_texture_float_linear")===!1&&(S.magFilter===je||S.magFilter===lo||S.magFilter===Nr||S.magFilter===si||S.minFilter===je||S.minFilter===lo||S.minFilter===Nr||S.minFilter===si)&&qt("WebGLRenderer: Unable to use linear filtering with floating point textures. OES_texture_float_linear not supported on this device."),i.texParameteri(R,i.TEXTURE_WRAP_S,L[S.wrapS]),i.texParameteri(R,i.TEXTURE_WRAP_T,L[S.wrapT]),(R===i.TEXTURE_3D||R===i.TEXTURE_2D_ARRAY)&&i.texParameteri(R,i.TEXTURE_WRAP_R,L[S.wrapR]),i.texParameteri(R,i.TEXTURE_MAG_FILTER,I[S.magFilter]),i.texParameteri(R,i.TEXTURE_MIN_FILTER,I[S.minFilter]),S.compareFunction&&(i.texParameteri(R,i.TEXTURE_COMPARE_MODE,i.COMPARE_REF_TO_TEXTURE),i.texParameteri(R,i.TEXTURE_COMPARE_FUNC,G[S.compareFunction])),t.has("EXT_texture_filter_anisotropic")===!0){if(S.magFilter===Ge||S.minFilter!==Nr&&S.minFilter!==si||S.type===xn&&t.has("OES_texture_float_linear")===!1)return;if(S.anisotropy>1||n.get(S).__currentAnisotropy){let rt=t.get("EXT_texture_filter_anisotropic");i.texParameterf(R,rt.TEXTURE_MAX_ANISOTROPY_EXT,Math.min(S.anisotropy,s.getMaxAnisotropy())),n.get(S).__currentAnisotropy=S.anisotropy}}}function ut(R,S){let rt=!1;R.__webglInit===void 0&&(R.__webglInit=!0,S.addEventListener("dispose",k),P(S));let St=S.source,yt=f.get(St);yt===void 0&&(yt={},f.set(St,yt));let Et=st(S);if(Et!==R.__cacheKey){yt[Et]===void 0&&(yt[Et]={texture:i.createTexture(),usedTimes:0},a.memory.textures++,rt=!0),yt[Et].usedTimes++;let Rt=yt[R.__cacheKey];Rt!==void 0&&(yt[R.__cacheKey].usedTimes--,Rt.usedTimes===0&&Y(S)),R.__cacheKey=Et,R.__webglTexture=yt[Et].texture}return rt}function E(R,S,rt){return Math.floor(Math.floor(R/rt)/S)}function $(R,S,rt,St){let Et=R.updateRanges;if(Et.length===0)e.texSubImage2D(i.TEXTURE_2D,0,0,0,S.width,S.height,rt,St,S.data);else{Et.sort((z,_t)=>z.start-_t.start);let Rt=0;for(let z=1;z<Et.length;z++){let _t=Et[Rt],dt=Et[z],It=_t.start+_t.count,At=E(dt.start,S.width,4),Ct=E(_t.start,S.width,4);dt.start<=It+1&&At===Ct&&E(dt.start+dt.count-1,S.width,4)===At?_t.count=Math.max(_t.count,dt.start+dt.count-_t.start):(++Rt,Et[Rt]=dt)}Et.length=Rt+1;let gt=e.getParameter(i.UNPACK_ROW_LENGTH),xt=e.getParameter(i.UNPACK_SKIP_PIXELS),Pt=e.getParameter(i.UNPACK_SKIP_ROWS);e.pixelStorei(i.UNPACK_ROW_LENGTH,S.width);for(let z=0,_t=Et.length;z<_t;z++){let dt=Et[z],It=Math.floor(dt.start/4),At=Math.ceil(dt.count/4),Ct=It%S.width,Wt=Math.floor(It/S.width),ee=At,Tt=1;e.pixelStorei(i.UNPACK_SKIP_PIXELS,Ct),e.pixelStorei(i.UNPACK_SKIP_ROWS,Wt),e.texSubImage2D(i.TEXTURE_2D,0,Ct,Wt,ee,Tt,rt,St,S.data)}R.clearUpdateRanges(),e.pixelStorei(i.UNPACK_ROW_LENGTH,gt),e.pixelStorei(i.UNPACK_SKIP_PIXELS,xt),e.pixelStorei(i.UNPACK_SKIP_ROWS,Pt)}}function B(R,S,rt){let St=i.TEXTURE_2D;(S.isDataArrayTexture||S.isCompressedArrayTexture)&&(St=i.TEXTURE_2D_ARRAY),S.isData3DTexture&&(St=i.TEXTURE_3D);let yt=ut(R,S),Et=S.source;e.bindTexture(St,R.__webglTexture,i.TEXTURE0+rt);let Rt=n.get(Et);if(Et.version!==Rt.__version||yt===!0){if(e.activeTexture(i.TEXTURE0+rt),(typeof ImageBitmap<"u"&&S.image instanceof ImageBitmap)===!1){let Tt=oe.getPrimaries(oe.workingColorSpace),Yt=S.colorSpace===ai?null:oe.getPrimaries(S.colorSpace),te=S.colorSpace===ai||Tt===Yt?i.NONE:i.BROWSER_DEFAULT_WEBGL;e.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,S.flipY),e.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,S.premultiplyAlpha),e.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,te)}e.pixelStorei(i.UNPACK_ALIGNMENT,S.unpackAlignment);let xt=v(S.image,!1,s.maxTextureSize);xt=Gt(S,xt);let Pt=r.convert(S.format,S.colorSpace),z=r.convert(S.type),_t=w(S.internalFormat,Pt,z,S.normalized,S.colorSpace,S.isVideoTexture);ht(St,S);let dt,It=S.mipmaps,At=S.isVideoTexture!==!0,Ct=Rt.__version===void 0||yt===!0,Wt=Et.dataReady,ee=M(S,xt);if(S.isDepthTexture)_t=x(S.format===ri,S.type),Ct&&(At?e.texStorage2D(i.TEXTURE_2D,1,_t,xt.width,xt.height):e.texImage2D(i.TEXTURE_2D,0,_t,xt.width,xt.height,0,Pt,z,null));else if(S.isDataTexture)if(It.length>0){At&&Ct&&e.texStorage2D(i.TEXTURE_2D,ee,_t,It[0].width,It[0].height);for(let Tt=0,Yt=It.length;Tt<Yt;Tt++)dt=It[Tt],At?Wt&&e.texSubImage2D(i.TEXTURE_2D,Tt,0,0,dt.width,dt.height,Pt,z,dt.data):e.texImage2D(i.TEXTURE_2D,Tt,_t,dt.width,dt.height,0,Pt,z,dt.data);S.generateMipmaps=!1}else At?(Ct&&e.texStorage2D(i.TEXTURE_2D,ee,_t,xt.width,xt.height),Wt&&$(S,xt,Pt,z)):e.texImage2D(i.TEXTURE_2D,0,_t,xt.width,xt.height,0,Pt,z,xt.data);else if(S.isCompressedTexture)if(S.isCompressedArrayTexture){At&&Ct&&e.texStorage3D(i.TEXTURE_2D_ARRAY,ee,_t,It[0].width,It[0].height,xt.depth);for(let Tt=0,Yt=It.length;Tt<Yt;Tt++)if(dt=It[Tt],S.format!==Dn)if(Pt!==null)if(At){if(Wt)if(S.layerUpdates.size>0){let te=Ec(dt.width,dt.height,S.format,S.type);for(let Nt of S.layerUpdates){let _e=dt.data.subarray(Nt*te/dt.data.BYTES_PER_ELEMENT,(Nt+1)*te/dt.data.BYTES_PER_ELEMENT);e.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,Tt,0,0,Nt,dt.width,dt.height,1,Pt,_e)}}else e.compressedTexSubImage3D(i.TEXTURE_2D_ARRAY,Tt,0,0,0,dt.width,dt.height,xt.depth,Pt,dt.data)}else e.compressedTexImage3D(i.TEXTURE_2D_ARRAY,Tt,_t,dt.width,dt.height,xt.depth,0,dt.data,0,0);else qt("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()");else At?Wt&&e.texSubImage3D(i.TEXTURE_2D_ARRAY,Tt,0,0,0,dt.width,dt.height,xt.depth,Pt,z,dt.data):e.texImage3D(i.TEXTURE_2D_ARRAY,Tt,_t,dt.width,dt.height,xt.depth,0,Pt,z,dt.data);S.layerUpdates.size>0&&S.clearLayerUpdates()}else{At&&Ct&&e.texStorage2D(i.TEXTURE_2D,ee,_t,It[0].width,It[0].height);for(let Tt=0,Yt=It.length;Tt<Yt;Tt++)dt=It[Tt],S.format!==Dn?Pt!==null?At?Wt&&e.compressedTexSubImage2D(i.TEXTURE_2D,Tt,0,0,dt.width,dt.height,Pt,dt.data):e.compressedTexImage2D(i.TEXTURE_2D,Tt,_t,dt.width,dt.height,0,dt.data):qt("WebGLRenderer: Attempt to load unsupported compressed texture format in .uploadTexture()"):At?Wt&&e.texSubImage2D(i.TEXTURE_2D,Tt,0,0,dt.width,dt.height,Pt,z,dt.data):e.texImage2D(i.TEXTURE_2D,Tt,_t,dt.width,dt.height,0,Pt,z,dt.data)}else if(S.isDataArrayTexture)if(At){if(Ct&&e.texStorage3D(i.TEXTURE_2D_ARRAY,ee,_t,xt.width,xt.height,xt.depth),Wt)if(S.layerUpdates.size>0){let Tt=Ec(xt.width,xt.height,S.format,S.type);for(let Yt of S.layerUpdates){let te=xt.data.subarray(Yt*Tt/xt.data.BYTES_PER_ELEMENT,(Yt+1)*Tt/xt.data.BYTES_PER_ELEMENT);e.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,Yt,xt.width,xt.height,1,Pt,z,te)}S.clearLayerUpdates()}else e.texSubImage3D(i.TEXTURE_2D_ARRAY,0,0,0,0,xt.width,xt.height,xt.depth,Pt,z,xt.data)}else e.texImage3D(i.TEXTURE_2D_ARRAY,0,_t,xt.width,xt.height,xt.depth,0,Pt,z,xt.data);else if(S.isData3DTexture)At?(Ct&&e.texStorage3D(i.TEXTURE_3D,ee,_t,xt.width,xt.height,xt.depth),Wt&&e.texSubImage3D(i.TEXTURE_3D,0,0,0,0,xt.width,xt.height,xt.depth,Pt,z,xt.data)):e.texImage3D(i.TEXTURE_3D,0,_t,xt.width,xt.height,xt.depth,0,Pt,z,xt.data);else if(S.isFramebufferTexture){if(Ct)if(At)e.texStorage2D(i.TEXTURE_2D,ee,_t,xt.width,xt.height);else{let Tt=xt.width,Yt=xt.height;for(let te=0;te<ee;te++)e.texImage2D(i.TEXTURE_2D,te,_t,Tt,Yt,0,Pt,z,null),Tt>>=1,Yt>>=1}}else if(S.isHTMLTexture){if("texElementImage2D"in i){let Tt=i.canvas;if(Tt.hasAttribute("layoutsubtree")||Tt.setAttribute("layoutsubtree","true"),xt.parentNode!==Tt){Tt.appendChild(xt),h.add(S),Tt.onpaint=Yt=>{let te=Yt.changedElements;for(let Nt of h)te.includes(Nt.image)&&(Nt.needsUpdate=!0)},Tt.requestPaint();return}if(i.texElementImage2D.length===3)i.texElementImage2D(i.TEXTURE_2D,i.RGBA8,xt);else{let te=i.RGBA,Nt=i.RGBA,_e=i.UNSIGNED_BYTE;i.texElementImage2D(i.TEXTURE_2D,0,te,Nt,_e,xt)}i.texParameteri(i.TEXTURE_2D,i.TEXTURE_MIN_FILTER,i.LINEAR),i.texParameteri(i.TEXTURE_2D,i.TEXTURE_WRAP_S,i.CLAMP_TO_EDGE),i.texParameteri(i.TEXTURE_2D,i.TEXTURE_WRAP_T,i.CLAMP_TO_EDGE)}}else if(It.length>0){if(At&&Ct){let Tt=$t(It[0]);e.texStorage2D(i.TEXTURE_2D,ee,_t,Tt.width,Tt.height)}for(let Tt=0,Yt=It.length;Tt<Yt;Tt++)dt=It[Tt],At?Wt&&e.texSubImage2D(i.TEXTURE_2D,Tt,0,0,Pt,z,dt):e.texImage2D(i.TEXTURE_2D,Tt,_t,Pt,z,dt);S.generateMipmaps=!1}else if(At){if(Ct){let Tt=$t(xt);e.texStorage2D(i.TEXTURE_2D,ee,_t,Tt.width,Tt.height)}Wt&&e.texSubImage2D(i.TEXTURE_2D,0,0,0,Pt,z,xt)}else e.texImage2D(i.TEXTURE_2D,0,_t,Pt,z,xt);b(S)&&S.mipmapsAutoUpdate===!0&&_(St),Rt.__version=Et.version,S.onUpdate&&S.onUpdate(S)}R.__version=S.version}function nt(R,S,rt){if(S.image.length!==6)return;let St=ut(R,S),yt=S.source;e.bindTexture(i.TEXTURE_CUBE_MAP,R.__webglTexture,i.TEXTURE0+rt);let Et=n.get(yt);if(yt.version!==Et.__version||St===!0){e.activeTexture(i.TEXTURE0+rt);let Rt=oe.getPrimaries(oe.workingColorSpace),gt=S.colorSpace===ai?null:oe.getPrimaries(S.colorSpace),xt=S.colorSpace===ai||Rt===gt?i.NONE:i.BROWSER_DEFAULT_WEBGL;e.pixelStorei(i.UNPACK_FLIP_Y_WEBGL,S.flipY),e.pixelStorei(i.UNPACK_PREMULTIPLY_ALPHA_WEBGL,S.premultiplyAlpha),e.pixelStorei(i.UNPACK_ALIGNMENT,S.unpackAlignment),e.pixelStorei(i.UNPACK_COLORSPACE_CONVERSION_WEBGL,xt);let Pt=S.isCompressedTexture||S.image[0].isCompressedTexture,z=S.image[0]&&S.image[0].isDataTexture,_t=[];for(let Nt=0;Nt<6;Nt++)!Pt&&!z?_t[Nt]=v(S.image[Nt],!0,s.maxCubemapSize):_t[Nt]=z?S.image[Nt].image:S.image[Nt],_t[Nt]=Gt(S,_t[Nt]);let dt=_t[0],It=r.convert(S.format,S.colorSpace),At=r.convert(S.type),Ct=w(S.internalFormat,It,At,S.normalized,S.colorSpace),Wt=S.isVideoTexture!==!0,ee=Et.__version===void 0||St===!0,Tt=yt.dataReady,Yt=M(S,dt);ht(i.TEXTURE_CUBE_MAP,S);let te;if(S.isDepthTexture){if(Ct=x(S.format===ri,S.type),ee)if(Wt)e.texStorage2D(i.TEXTURE_CUBE_MAP,1,Ct,dt.width,dt.height);else for(let Nt=0;Nt<6;Nt++)e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,0,Ct,dt.width,dt.height,0,It,At,null)}else if(Pt){Wt&&ee&&e.texStorage2D(i.TEXTURE_CUBE_MAP,Yt,Ct,dt.width,dt.height);for(let Nt=0;Nt<6;Nt++){te=_t[Nt].mipmaps;for(let _e=0;_e<te.length;_e++){let Be=te[_e];S.format!==Dn?It!==null?Wt?Tt&&e.compressedTexSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,_e,0,0,Be.width,Be.height,It,Be.data):e.compressedTexImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,_e,Ct,Be.width,Be.height,0,Be.data):qt("WebGLRenderer: Attempt to load unsupported compressed texture format in .setTextureCube()"):Wt?Tt&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,_e,0,0,Be.width,Be.height,It,At,Be.data):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,_e,Ct,Be.width,Be.height,0,It,At,Be.data)}}}else{if(te=S.mipmaps,Wt&&ee){te.length>0&&Yt++;let Nt=$t(_t[0]);e.texStorage2D(i.TEXTURE_CUBE_MAP,Yt,Ct,Nt.width,Nt.height)}for(let Nt=0;Nt<6;Nt++)if(z){Wt?Tt&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,0,0,0,_t[Nt].width,_t[Nt].height,It,At,_t[Nt].data):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,0,Ct,_t[Nt].width,_t[Nt].height,0,It,At,_t[Nt].data);for(let _e=0;_e<te.length;_e++){let ci=te[_e].image[Nt].image;Wt?Tt&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,_e+1,0,0,ci.width,ci.height,It,At,ci.data):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,_e+1,Ct,ci.width,ci.height,0,It,At,ci.data)}}else{Wt?Tt&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,0,0,0,It,At,_t[Nt]):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,0,Ct,It,At,_t[Nt]);for(let _e=0;_e<te.length;_e++){let Be=te[_e];Wt?Tt&&e.texSubImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,_e+1,0,0,It,At,Be.image[Nt]):e.texImage2D(i.TEXTURE_CUBE_MAP_POSITIVE_X+Nt,_e+1,Ct,It,At,Be.image[Nt])}}}b(S)&&S.mipmapsAutoUpdate===!0&&_(i.TEXTURE_CUBE_MAP),Et.__version=yt.version,S.onUpdate&&S.onUpdate(S)}R.__version=S.version}function it(R,S,rt,St,yt,Et){let Rt=r.convert(rt.format,rt.colorSpace),gt=r.convert(rt.type),xt=w(rt.internalFormat,Rt,gt,rt.normalized,rt.colorSpace),Pt=n.get(S),z=n.get(rt);if(z.__renderTarget=S,!Pt.__hasExternalTextures){let _t=Math.max(1,S.width>>Et),dt=Math.max(1,S.height>>Et);yt===i.TEXTURE_3D||yt===i.TEXTURE_2D_ARRAY?e.texImage3D(yt,Et,xt,_t,dt,S.depth,0,Rt,gt,null):e.texImage2D(yt,Et,xt,_t,dt,0,Rt,gt,null)}e.bindFramebuffer(i.FRAMEBUFFER,R),Mt(S)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,St,yt,z.__webglTexture,0,se(S)):(yt===i.TEXTURE_2D||yt>=i.TEXTURE_CUBE_MAP_POSITIVE_X&&yt<=i.TEXTURE_CUBE_MAP_NEGATIVE_Z)&&i.framebufferTexture2D(i.FRAMEBUFFER,St,yt,z.__webglTexture,Et),e.bindFramebuffer(i.FRAMEBUFFER,null)}function pt(R,S,rt){if(i.bindRenderbuffer(i.RENDERBUFFER,R),S.depthBuffer){let St=S.depthTexture,yt=St&&St.isDepthTexture?St.type:null,Et=x(S.stencilBuffer,yt),Rt=S.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;Mt(S)?o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,se(S),Et,S.width,S.height):rt?i.renderbufferStorageMultisample(i.RENDERBUFFER,se(S),Et,S.width,S.height):i.renderbufferStorage(i.RENDERBUFFER,Et,S.width,S.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,Rt,i.RENDERBUFFER,R)}else{let St=S.textures;for(let yt=0;yt<St.length;yt++){let Et=St[yt],Rt=r.convert(Et.format,Et.colorSpace),gt=r.convert(Et.type),xt=w(Et.internalFormat,Rt,gt,Et.normalized,Et.colorSpace);Mt(S)?o.renderbufferStorageMultisampleEXT(i.RENDERBUFFER,se(S),xt,S.width,S.height):rt?i.renderbufferStorageMultisample(i.RENDERBUFFER,se(S),xt,S.width,S.height):i.renderbufferStorage(i.RENDERBUFFER,xt,S.width,S.height)}}i.bindRenderbuffer(i.RENDERBUFFER,null)}function mt(R,S,rt){let St=S.isWebGLCubeRenderTarget===!0;if(e.bindFramebuffer(i.FRAMEBUFFER,R),!(S.depthTexture&&S.depthTexture.isDepthTexture))throw new Error("THREE.WebGLTextures: renderTarget.depthTexture must be an instance of THREE.DepthTexture.");let yt=n.get(S.depthTexture);yt.__renderTarget=S;let Et=St?S.depthTexture.image[0]:S.depthTexture.image;(!yt.__webglTexture||Et.width!==S.width||Et.height!==S.height)&&(Et.width=S.width,Et.height=S.height,S.depthTexture.needsUpdate=!0),St?N(S.depthTexture,0):U(S.depthTexture,0);let Rt=yt.__webglTexture,gt=se(S),xt=St?i.TEXTURE_CUBE_MAP_POSITIVE_X+rt:i.TEXTURE_2D,Pt=S.depthTexture.format===ri?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;if(S.depthTexture.format===Kn)Mt(S)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,Pt,xt,Rt,0,gt):i.framebufferTexture2D(i.FRAMEBUFFER,Pt,xt,Rt,0);else if(S.depthTexture.format===ri)Mt(S)?o.framebufferTexture2DMultisampleEXT(i.FRAMEBUFFER,Pt,xt,Rt,0,gt):i.framebufferTexture2D(i.FRAMEBUFFER,Pt,xt,Rt,0);else throw new Error("THREE.WebGLTextures: Unknown depthTexture format.")}function X(R){let S=n.get(R),rt=R.isWebGLCubeRenderTarget===!0;if(S.__boundDepthTexture!==R.depthTexture){let St=R.depthTexture;if(S.__depthDisposeCallback&&S.__depthDisposeCallback(),St){let yt=()=>{delete S.__boundDepthTexture,delete S.__depthDisposeCallback,St.removeEventListener("dispose",yt)};St.addEventListener("dispose",yt),S.__depthDisposeCallback=yt}S.__boundDepthTexture=St}if(R.depthTexture&&!S.__autoAllocateDepthBuffer)if(rt)for(let St=0;St<6;St++)mt(S.__webglFramebuffer[St],R,St);else R.texture.mipmaps.length>0?mt(S.__webglFramebuffer[0],R,0):mt(S.__webglFramebuffer,R,0);else if(rt){S.__webglDepthbuffer=[];for(let St=0;St<6;St++)if(e.bindFramebuffer(i.FRAMEBUFFER,S.__webglFramebuffer[St]),S.__webglDepthbuffer[St]===void 0)S.__webglDepthbuffer[St]=i.createRenderbuffer(),pt(S.__webglDepthbuffer[St],R,!1);else{let yt=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Et=S.__webglDepthbuffer[St];i.bindRenderbuffer(i.RENDERBUFFER,Et),i.framebufferRenderbuffer(i.FRAMEBUFFER,yt,i.RENDERBUFFER,Et)}}else if(R.texture.mipmaps.length>0?e.bindFramebuffer(i.FRAMEBUFFER,S.__webglFramebuffer[0]):e.bindFramebuffer(i.FRAMEBUFFER,S.__webglFramebuffer),S.__webglDepthbuffer===void 0)S.__webglDepthbuffer=i.createRenderbuffer(),pt(S.__webglDepthbuffer,R,!1);else{let yt=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Et=S.__webglDepthbuffer;i.bindRenderbuffer(i.RENDERBUFFER,Et),i.framebufferRenderbuffer(i.FRAMEBUFFER,yt,i.RENDERBUFFER,Et)}e.bindFramebuffer(i.FRAMEBUFFER,null)}function Z(R,S,rt){let St=n.get(R);S!==void 0&&it(St.__webglFramebuffer,R,R.texture,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,0),rt!==void 0&&X(R)}function bt(R){let S=R.texture,rt=n.get(R),St=n.get(S);R.addEventListener("dispose",O),P(R);let yt=R.textures,Et=R.isWebGLCubeRenderTarget===!0,Rt=yt.length>1;if(Rt||(St.__webglTexture===void 0&&(St.__webglTexture=i.createTexture()),St.__version=S.version,a.memory.textures++),Et){rt.__webglFramebuffer=[];for(let gt=0;gt<6;gt++)if(S.mipmaps.length>0){rt.__webglFramebuffer[gt]=[];for(let xt=0;xt<S.mipmaps.length;xt++)rt.__webglFramebuffer[gt][xt]=i.createFramebuffer()}else rt.__webglFramebuffer[gt]=i.createFramebuffer()}else{if(S.mipmaps.length>0){rt.__webglFramebuffer=[];for(let gt=0;gt<S.mipmaps.length;gt++)rt.__webglFramebuffer[gt]=i.createFramebuffer()}else rt.__webglFramebuffer=i.createFramebuffer();if(Rt)for(let gt=0,xt=yt.length;gt<xt;gt++){let Pt=n.get(yt[gt]);Pt.__webglTexture===void 0&&(Pt.__webglTexture=i.createTexture(),a.memory.textures++)}if(R.samples>0&&Mt(R)===!1){rt.__webglMultisampledFramebuffer=i.createFramebuffer(),rt.__webglColorRenderbuffer=[],e.bindFramebuffer(i.FRAMEBUFFER,rt.__webglMultisampledFramebuffer);for(let gt=0;gt<yt.length;gt++){let xt=yt[gt];rt.__webglColorRenderbuffer[gt]=i.createRenderbuffer(),i.bindRenderbuffer(i.RENDERBUFFER,rt.__webglColorRenderbuffer[gt]);let Pt=r.convert(xt.format,xt.colorSpace),z=r.convert(xt.type),_t=w(xt.internalFormat,Pt,z,xt.normalized,xt.colorSpace,R.isXRRenderTarget===!0),dt=se(R);i.renderbufferStorageMultisample(i.RENDERBUFFER,dt,_t,R.width,R.height),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+gt,i.RENDERBUFFER,rt.__webglColorRenderbuffer[gt])}i.bindRenderbuffer(i.RENDERBUFFER,null),R.depthBuffer&&(rt.__webglDepthRenderbuffer=i.createRenderbuffer(),pt(rt.__webglDepthRenderbuffer,R,!0)),e.bindFramebuffer(i.FRAMEBUFFER,null)}}if(Et){e.bindTexture(i.TEXTURE_CUBE_MAP,St.__webglTexture),ht(i.TEXTURE_CUBE_MAP,S);for(let gt=0;gt<6;gt++)if(S.mipmaps.length>0)for(let xt=0;xt<S.mipmaps.length;xt++)it(rt.__webglFramebuffer[gt][xt],R,S,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+gt,xt);else it(rt.__webglFramebuffer[gt],R,S,i.COLOR_ATTACHMENT0,i.TEXTURE_CUBE_MAP_POSITIVE_X+gt,0);b(S)?_(i.TEXTURE_CUBE_MAP):S.mipmaps.length>0&&i.texParameteri(i.TEXTURE_CUBE_MAP,i.TEXTURE_MAX_LEVEL,S.mipmaps.length-1),e.unbindTexture()}else if(Rt){for(let gt=0,xt=yt.length;gt<xt;gt++){let Pt=yt[gt],z=n.get(Pt),_t=i.TEXTURE_2D;(R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(_t=R.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),e.bindTexture(_t,z.__webglTexture),ht(_t,Pt),it(rt.__webglFramebuffer,R,Pt,i.COLOR_ATTACHMENT0+gt,_t,0),b(Pt)&&_(_t)}e.unbindTexture()}else{let gt=i.TEXTURE_2D;if((R.isWebGL3DRenderTarget||R.isWebGLArrayRenderTarget)&&(gt=R.isWebGL3DRenderTarget?i.TEXTURE_3D:i.TEXTURE_2D_ARRAY),e.bindTexture(gt,St.__webglTexture),ht(gt,S),S.mipmaps.length>0)for(let xt=0;xt<S.mipmaps.length;xt++)it(rt.__webglFramebuffer[xt],R,S,i.COLOR_ATTACHMENT0,gt,xt);else it(rt.__webglFramebuffer,R,S,i.COLOR_ATTACHMENT0,gt,0);b(S)?_(gt):S.mipmaps.length>0&&i.texParameteri(gt,i.TEXTURE_MAX_LEVEL,S.mipmaps.length-1),e.unbindTexture()}R.depthBuffer&&X(R)}function H(R){let S=R.textures;for(let rt=0,St=S.length;rt<St;rt++){let yt=S[rt];if(b(yt)&&yt.mipmapsAutoUpdate===!0){let Et=T(R),Rt=n.get(yt).__webglTexture;e.bindTexture(Et,Rt),_(Et),e.unbindTexture()}}}let Qt=[];function jt(R){if(R.samples>0){if(Mt(R)===!1){let S=R.textures,rt=R.width,St=R.height,yt=i.COLOR_BUFFER_BIT,Et=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT,Rt=n.get(R),gt=S.length>1;if(gt)for(let Pt=0;Pt<S.length;Pt++)e.bindFramebuffer(i.FRAMEBUFFER,Rt.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+Pt,i.RENDERBUFFER,null),e.bindFramebuffer(i.FRAMEBUFFER,Rt.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+Pt,i.TEXTURE_2D,null,0);e.bindFramebuffer(i.READ_FRAMEBUFFER,Rt.__webglMultisampledFramebuffer),R.texture.mipmaps.length>0?e.bindFramebuffer(i.DRAW_FRAMEBUFFER,Rt.__webglFramebuffer[0]):e.bindFramebuffer(i.DRAW_FRAMEBUFFER,Rt.__webglFramebuffer);for(let Pt=0;Pt<S.length;Pt++){if(R.resolveDepthBuffer&&(R.depthBuffer&&(yt|=i.DEPTH_BUFFER_BIT),R.stencilBuffer&&R.resolveStencilBuffer&&(yt|=i.STENCIL_BUFFER_BIT)),gt){i.framebufferRenderbuffer(i.READ_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.RENDERBUFFER,Rt.__webglColorRenderbuffer[Pt]);let z=n.get(S[Pt]).__webglTexture;i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0,i.TEXTURE_2D,z,0)}i.blitFramebuffer(0,0,rt,St,0,0,rt,St,yt,i.NEAREST),c===!0&&(Qt.length=0,Qt.push(i.COLOR_ATTACHMENT0+Pt),R.depthBuffer&&R.storeMultisampledDepthBuffer===!1&&Qt.push(Et),i.invalidateFramebuffer(i.READ_FRAMEBUFFER,Qt))}if(e.bindFramebuffer(i.READ_FRAMEBUFFER,null),e.bindFramebuffer(i.DRAW_FRAMEBUFFER,null),gt)for(let Pt=0;Pt<S.length;Pt++){e.bindFramebuffer(i.FRAMEBUFFER,Rt.__webglMultisampledFramebuffer),i.framebufferRenderbuffer(i.FRAMEBUFFER,i.COLOR_ATTACHMENT0+Pt,i.RENDERBUFFER,Rt.__webglColorRenderbuffer[Pt]);let z=n.get(S[Pt]).__webglTexture;e.bindFramebuffer(i.FRAMEBUFFER,Rt.__webglFramebuffer),i.framebufferTexture2D(i.DRAW_FRAMEBUFFER,i.COLOR_ATTACHMENT0+Pt,i.TEXTURE_2D,z,0)}e.bindFramebuffer(i.DRAW_FRAMEBUFFER,Rt.__webglMultisampledFramebuffer)}else if(R.depthBuffer&&R.storeMultisampledDepthBuffer===!1&&c){let S=R.stencilBuffer?i.DEPTH_STENCIL_ATTACHMENT:i.DEPTH_ATTACHMENT;i.invalidateFramebuffer(i.DRAW_FRAMEBUFFER,[S])}}}function se(R){return Math.min(s.maxSamples,R.samples)}function Mt(R){let S=n.get(R);return R.samples>0&&t.has("WEBGL_multisampled_render_to_texture")===!0&&S.__useRenderToTexture!==!1}function fe(R){let S=a.render.frame;u.get(R)!==S&&(u.set(R,S),R.update())}function Gt(R,S){let rt=R.colorSpace,St=R.format,yt=R.type;return R.isCompressedTexture===!0||R.isVideoTexture===!0||rt!==ir&&rt!==ai&&(oe.getTransfer(rt)===ge?(St!==Dn||yt!==nn)&&qt("WebGLTextures: sRGB encoded textures have to use RGBAFormat and UnsignedByteType."):Zt("WebGLTextures: Unsupported texture color space:",rt)),S}function $t(R){return typeof HTMLImageElement<"u"&&R instanceof HTMLImageElement?(l.width=R.naturalWidth||R.width,l.height=R.naturalHeight||R.height):typeof VideoFrame<"u"&&R instanceof VideoFrame?(l.width=R.displayWidth,l.height=R.displayHeight):(l.width=R.width,l.height=R.height),l}this.allocateTextureUnit=wt,this.resetTextureUnits=W,this.getTextureUnits=tt,this.setTextureUnits=vt,this.setTexture2D=U,this.setTexture2DArray=et,this.setTexture3D=V,this.setTextureCube=N,this.rebindTextures=Z,this.setupRenderTarget=bt,this.updateRenderTargetMipmap=H,this.updateMultisampleRenderTarget=jt,this.setupDepthRenderbuffer=X,this.setupFrameBufferTexture=it,this.useMultisampledRTT=Mt,this.dispose=K}function a_(i,t){function e(n,s=ai){let r,a=oe.getTransfer(s);if(n===nn)return i.UNSIGNED_BYTE;if(n===ho)return i.UNSIGNED_SHORT_4_4_4_4;if(n===uo)return i.UNSIGNED_SHORT_5_5_5_1;if(n===uc)return i.UNSIGNED_INT_5_9_9_9_REV;if(n===dc)return i.UNSIGNED_INT_10F_11F_11F_REV;if(n===cc)return i.BYTE;if(n===hc)return i.SHORT;if(n===Us)return i.UNSIGNED_SHORT;if(n===co)return i.INT;if(n===In)return i.UNSIGNED_INT;if(n===xn)return i.FLOAT;if(n===Ln)return i.HALF_FLOAT;if(n===fc)return i.ALPHA;if(n===pc)return i.RGB;if(n===Dn)return i.RGBA;if(n===Kn)return i.DEPTH_COMPONENT;if(n===ri)return i.DEPTH_STENCIL;if(n===fo)return i.RED;if(n===po)return i.RED_INTEGER;if(n===Ti)return i.RG;if(n===mo)return i.RG_INTEGER;if(n===go)return i.RGBA_INTEGER;if(n===Ur||n===Fr||n===Or||n===Br)if(a===ge)if(r=t.get("WEBGL_compressed_texture_s3tc_srgb"),r!==null){if(n===Ur)return r.COMPRESSED_SRGB_S3TC_DXT1_EXT;if(n===Fr)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT;if(n===Or)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT;if(n===Br)return r.COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT}else return null;else if(r=t.get("WEBGL_compressed_texture_s3tc"),r!==null){if(n===Ur)return r.COMPRESSED_RGB_S3TC_DXT1_EXT;if(n===Fr)return r.COMPRESSED_RGBA_S3TC_DXT1_EXT;if(n===Or)return r.COMPRESSED_RGBA_S3TC_DXT3_EXT;if(n===Br)return r.COMPRESSED_RGBA_S3TC_DXT5_EXT}else return null;if(n===xo||n===_o||n===yo||n===vo)if(r=t.get("WEBGL_compressed_texture_pvrtc"),r!==null){if(n===xo)return r.COMPRESSED_RGB_PVRTC_4BPPV1_IMG;if(n===_o)return r.COMPRESSED_RGB_PVRTC_2BPPV1_IMG;if(n===yo)return r.COMPRESSED_RGBA_PVRTC_4BPPV1_IMG;if(n===vo)return r.COMPRESSED_RGBA_PVRTC_2BPPV1_IMG}else return null;if(n===bo||n===Mo||n===So||n===wo||n===To||n===kr||n===Ao)if(r=t.get("WEBGL_compressed_texture_etc"),r!==null){if(n===bo||n===Mo)return a===ge?r.COMPRESSED_SRGB8_ETC2:r.COMPRESSED_RGB8_ETC2;if(n===So)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ETC2_EAC:r.COMPRESSED_RGBA8_ETC2_EAC;if(n===wo)return r.COMPRESSED_R11_EAC;if(n===To)return r.COMPRESSED_SIGNED_R11_EAC;if(n===kr)return r.COMPRESSED_RG11_EAC;if(n===Ao)return r.COMPRESSED_SIGNED_RG11_EAC}else return null;if(n===Eo||n===Co||n===Ro||n===Po||n===Io||n===Lo||n===Do||n===No||n===Uo||n===Fo||n===Oo||n===Bo||n===ko||n===zo)if(r=t.get("WEBGL_compressed_texture_astc"),r!==null){if(n===Eo)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_4x4_KHR:r.COMPRESSED_RGBA_ASTC_4x4_KHR;if(n===Co)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x4_KHR:r.COMPRESSED_RGBA_ASTC_5x4_KHR;if(n===Ro)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_5x5_KHR:r.COMPRESSED_RGBA_ASTC_5x5_KHR;if(n===Po)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x5_KHR:r.COMPRESSED_RGBA_ASTC_6x5_KHR;if(n===Io)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_6x6_KHR:r.COMPRESSED_RGBA_ASTC_6x6_KHR;if(n===Lo)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x5_KHR:r.COMPRESSED_RGBA_ASTC_8x5_KHR;if(n===Do)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x6_KHR:r.COMPRESSED_RGBA_ASTC_8x6_KHR;if(n===No)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_8x8_KHR:r.COMPRESSED_RGBA_ASTC_8x8_KHR;if(n===Uo)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x5_KHR:r.COMPRESSED_RGBA_ASTC_10x5_KHR;if(n===Fo)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x6_KHR:r.COMPRESSED_RGBA_ASTC_10x6_KHR;if(n===Oo)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x8_KHR:r.COMPRESSED_RGBA_ASTC_10x8_KHR;if(n===Bo)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_10x10_KHR:r.COMPRESSED_RGBA_ASTC_10x10_KHR;if(n===ko)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x10_KHR:r.COMPRESSED_RGBA_ASTC_12x10_KHR;if(n===zo)return a===ge?r.COMPRESSED_SRGB8_ALPHA8_ASTC_12x12_KHR:r.COMPRESSED_RGBA_ASTC_12x12_KHR}else return null;if(n===Go||n===Vo||n===Ho)if(r=t.get("EXT_texture_compression_bptc"),r!==null){if(n===Go)return a===ge?r.COMPRESSED_SRGB_ALPHA_BPTC_UNORM_EXT:r.COMPRESSED_RGBA_BPTC_UNORM_EXT;if(n===Vo)return r.COMPRESSED_RGB_BPTC_SIGNED_FLOAT_EXT;if(n===Ho)return r.COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT_EXT}else return null;if(n===Wo||n===Xo||n===zr||n===qo)if(r=t.get("EXT_texture_compression_rgtc"),r!==null){if(n===Wo)return r.COMPRESSED_RED_RGTC1_EXT;if(n===Xo)return r.COMPRESSED_SIGNED_RED_RGTC1_EXT;if(n===zr)return r.COMPRESSED_RED_GREEN_RGTC2_EXT;if(n===qo)return r.COMPRESSED_SIGNED_RED_GREEN_RGTC2_EXT}else return null;return n===$i?i.UNSIGNED_INT_24_8:i[n]!==void 0?i[n]:null}return{convert:e}}var o_=`
void main() {

	gl_Position = vec4( position, 1.0 );

}`,l_=`
uniform sampler2DArray depthColor;
uniform float depthWidth;
uniform float depthHeight;

void main() {

	vec2 coord = vec2( gl_FragCoord.x / depthWidth, gl_FragCoord.y / depthHeight );

	if ( coord.x >= 1.0 ) {

		gl_FragDepth = texture( depthColor, vec3( coord.x - 1.0, coord.y, 1 ) ).r;

	} else {

		gl_FragDepth = texture( depthColor, vec3( coord.x, coord.y, 0 ) ).r;

	}

}`,Fc=class{constructor(){this.texture=null,this.mesh=null,this.depthNear=0,this.depthFar=0}init(t,e){if(this.texture===null){let n=new mr(t.texture);(t.depthNear!==e.depthNear||t.depthFar!==e.depthFar)&&(this.depthNear=t.depthNear,this.depthFar=t.depthFar),this.texture=n}}getMesh(t){if(this.texture!==null&&this.mesh===null){let e=t.cameras[0].viewport,n=new dn({vertexShader:o_,fragmentShader:l_,uniforms:{depthColor:{value:this.texture},depthWidth:{value:e.z},depthHeight:{value:e.w}}});this.mesh=new he(new ni(20,20),n)}return this.mesh}reset(){this.texture=null,this.mesh=null}getDepthTexture(){return this.texture}},Oc=class extends Sn{constructor(t,e){super();let n=this,s=null,r=1,a=null,o="local-floor",c=1,l=null,u=null,h=null,d=null,f=null,m=null,y=typeof XRWebGLBinding<"u",g=new Fc,p={},v=e.getContextAttributes(),b=null,_=null,T=[],w=[],x=new ft,M=null,P=null,C=new Ke;C.viewport=new Se;let k=new Ke;k.viewport=new Se;let O=[C,k],j=new ro,D=null,K=null;this.cameraAutoUpdate=!0,this.enabled=!1,this.isPresenting=!1,this.getController=function(st){let U=T[st];return U===void 0&&(U=new Ms,T[st]=U),U.getTargetRaySpace()},this.getControllerGrip=function(st){let U=T[st];return U===void 0&&(U=new Ms,T[st]=U),U.getGripSpace()},this.getHand=function(st){let U=T[st];return U===void 0&&(U=new Ms,T[st]=U),U.getHandSpace()};function Q(st){let U=w.indexOf(st.inputSource);if(U===-1)return;let et=T[U];et!==void 0&&(et.update(st.inputSource,st.frame,l||a),et.dispatchEvent({type:st.type,data:st.inputSource}))}function Y(){s.removeEventListener("select",Q),s.removeEventListener("selectstart",Q),s.removeEventListener("selectend",Q),s.removeEventListener("squeeze",Q),s.removeEventListener("squeezestart",Q),s.removeEventListener("squeezeend",Q),s.removeEventListener("end",Y),s.removeEventListener("inputsourceschange",J);for(let st=0;st<T.length;st++){let U=w[st];U!==null&&(w[st]=null,T[st].disconnect(U))}D=null,K=null,g.reset();for(let st in p)delete p[st];if(t.setRenderTarget(b),f=null,d=null,h=null,s=null,_=null,wt.stop(),n.isPresenting=!1,t.setPixelRatio(M),t.setSize(x.width,x.height,!1),P!==null){let st=P.camera;st.fov=P.fov,st.zoom=P.zoom,st.updateProjectionMatrix(),P=null}n.dispatchEvent({type:"sessionend"})}this.setFramebufferScaleFactor=function(st){r=st,n.isPresenting===!0&&qt("WebXRManager: Cannot change framebuffer scale while presenting.")},this.setReferenceSpaceType=function(st){o=st,n.isPresenting===!0&&qt("WebXRManager: Cannot change reference space type while presenting.")},this.getReferenceSpace=function(){return l||a},this.setReferenceSpace=function(st){l=st},this.getBaseLayer=function(){return d!==null?d:f},this.getBinding=function(){return h===null&&y&&(h=new XRWebGLBinding(s,e)),h},this.getFrame=function(){return m},this.getSession=function(){return s},this.setSession=async function(st){if(s=st,s!==null){if(b=t.getRenderTarget(),s.addEventListener("select",Q),s.addEventListener("selectstart",Q),s.addEventListener("selectend",Q),s.addEventListener("squeeze",Q),s.addEventListener("squeezestart",Q),s.addEventListener("squeezeend",Q),s.addEventListener("end",Y),s.addEventListener("inputsourceschange",J),v.xrCompatible!==!0&&await e.makeXRCompatible(),M=t.getPixelRatio(),t.getSize(x),y&&"createProjectionLayer"in XRWebGLBinding.prototype){let et=null,V=null,N=null;v.depth&&(N=v.stencil?e.DEPTH24_STENCIL8:e.DEPTH_COMPONENT24,et=v.stencil?ri:Kn,V=v.stencil?$i:In);let L={colorFormat:e.RGBA8,depthFormat:N,scaleFactor:r};h=this.getBinding(),d=h.createProjectionLayer(L),s.updateRenderState({layers:[d]}),t.setPixelRatio(1),t.setSize(d.textureWidth,d.textureHeight,!1),_=new cn(d.textureWidth,d.textureHeight,{format:Dn,type:nn,depthTexture:new ei(d.textureWidth,d.textureHeight,V,void 0,void 0,void 0,void 0,void 0,void 0,et),stencilBuffer:v.stencil,colorSpace:t.outputColorSpace,samples:v.antialias?4:0,resolveDepthBuffer:d.ignoreDepthValues===!1,resolveStencilBuffer:d.ignoreDepthValues===!1,storeMultisampledDepthBuffer:d.ignoreDepthValues===!1,storeMultisampledStencilBuffer:d.ignoreDepthValues===!1})}else{let et={antialias:v.antialias,alpha:!0,depth:v.depth,stencil:v.stencil,framebufferScaleFactor:r};f=new XRWebGLLayer(s,e,et),s.updateRenderState({baseLayer:f}),t.setPixelRatio(1),t.setSize(f.framebufferWidth,f.framebufferHeight,!1),_=new cn(f.framebufferWidth,f.framebufferHeight,{format:Dn,type:nn,colorSpace:t.outputColorSpace,stencilBuffer:v.stencil,resolveDepthBuffer:f.ignoreDepthValues===!1,resolveStencilBuffer:f.ignoreDepthValues===!1,storeMultisampledDepthBuffer:f.ignoreDepthValues===!1,storeMultisampledStencilBuffer:f.ignoreDepthValues===!1})}_.isXRRenderTarget=!0,this.setFoveation(c),l=null,a=await s.requestReferenceSpace(o),wt.setContext(s),wt.start(),n.isPresenting=!0,n.dispatchEvent({type:"sessionstart"})}},this.getEnvironmentBlendMode=function(){if(s!==null)return s.environmentBlendMode},this.getDepthTexture=function(){return g.getDepthTexture()};function J(st){for(let U=0;U<st.removed.length;U++){let et=st.removed[U],V=w.indexOf(et);V>=0&&(w[V]=null,T[V].disconnect(et))}for(let U=0;U<st.added.length;U++){let et=st.added[U],V=w.indexOf(et);if(V===-1){for(let L=0;L<T.length;L++)if(L>=w.length){w.push(et),V=L;break}else if(w[L]===null){w[L]=et,V=L;break}if(V===-1)break}let N=T[V];N&&N.connect(et)}}function at(st,U){U===null?st.matrixWorld.copy(st.matrix):st.matrixWorld.multiplyMatrices(U.matrixWorld,st.matrix),st.matrixWorldInverse.copy(st.matrixWorld).invert()}this.updateCamera=function(st){if(s===null)return;let U=st.near,et=st.far;g.texture!==null&&(g.depthNear>0&&(U=g.depthNear),g.depthFar>0&&(et=g.depthFar)),j.near=k.near=C.near=U,j.far=k.far=C.far=et,(D!==j.near||K!==j.far)&&(s.updateRenderState({depthNear:j.near,depthFar:j.far}),D=j.near,K=j.far),j.layers.mask=st.layers.mask|6,C.layers.mask=j.layers.mask&-5,k.layers.mask=j.layers.mask&-3;let V=st.parent,N=j.cameras;at(j,V);for(let L=0;L<N.length;L++)at(N[L],V);j.projectionMatrix.copy(C.projectionMatrix),j.projectionMatrixInverse.copy(C.projectionMatrixInverse),P===null&&st.isPerspectiveCamera&&(P={camera:st,fov:st.fov,zoom:st.zoom}),W(st,j,V)};function W(st,U,et){et===null?st.matrix.copy(U.matrixWorld):(st.matrix.copy(et.matrixWorld),st.matrix.invert(),st.matrix.multiply(U.matrixWorld)),st.matrix.decompose(st.position,st.quaternion,st.scale),st.updateMatrixWorld(!0),st.projectionMatrix.copy(U.projectionMatrix),st.projectionMatrixInverse.copy(U.projectionMatrixInverse),st.isPerspectiveCamera&&(st.fov=ys*2*Math.atan(1/st.projectionMatrix.elements[5]),st.zoom=1)}this.getCamera=function(){return j},this.getFoveation=function(){if(!(d===null&&f===null))return c},this.setFoveation=function(st){c=st,d!==null&&(d.fixedFoveation=st),f!==null&&f.fixedFoveation!==void 0&&(f.fixedFoveation=st)},this.hasDepthSensing=function(){return g.texture!==null},this.getDepthSensingMesh=function(){return g.getMesh(j)},this.getCameraTexture=function(st){return p[st]};let tt=null;function vt(st,U){if(u=U.getViewerPose(l||a),m=U,u!==null){let et=u.views;f!==null&&(t.setRenderTargetFramebuffer(_,f.framebuffer),t.setRenderTarget(_));let V=!1;et.length!==j.cameras.length&&(j.cameras.length=0,V=!0);for(let G=0;G<et.length;G++){let ht=et[G],ut=null;if(f!==null)ut=f.getViewport(ht);else{let $=h.getViewSubImage(d,ht);ut=$.viewport,G===0&&(t.setRenderTargetTextures(_,$.colorTexture,$.depthStencilTexture),t.setRenderTarget(_))}let E=O[G];E===void 0&&(E=new Ke,E.layers.enable(G),E.viewport=new Se,O[G]=E),E.matrix.fromArray(ht.transform.matrix),E.matrix.decompose(E.position,E.quaternion,E.scale),E.projectionMatrix.fromArray(ht.projectionMatrix),E.projectionMatrixInverse.copy(E.projectionMatrix).invert(),E.viewport.set(ut.x,ut.y,ut.width,ut.height),G===0&&(j.matrix.copy(E.matrix),j.matrix.decompose(j.position,j.quaternion,j.scale)),V===!0&&j.cameras.push(E)}let N=s.enabledFeatures;if(N&&N.includes("depth-sensing")&&s.depthUsage=="gpu-optimized"&&y){h=n.getBinding();let G=h.getDepthInformation(et[0]);G&&G.isValid&&G.texture&&g.init(G,s.renderState)}if(N&&N.includes("camera-access")&&y){t.state.unbindTexture(),h=n.getBinding();for(let G=0;G<et.length;G++){let ht=et[G].camera;if(ht){let ut=p[ht];ut||(ut=new mr,p[ht]=ut);let E=h.getCameraImage(ht);ut.sourceTexture=E}}}}for(let et=0;et<T.length;et++){let V=w[et],N=T[et];V!==null&&N!==void 0&&N.update(V,U,l||a)}tt&&tt(st,U),U.detectedPlanes&&n.dispatchEvent({type:"planesdetected",data:U}),m=null}let wt=new ld;wt.setAnimationLoop(vt),this.setAnimationLoop=function(st){tt=st},this.dispose=function(){}}},c_=new xe,_d=new Xt;_d.set(-1,0,0,0,1,0,0,0,1);function h_(i,t){function e(g,p){g.matrixAutoUpdate===!0&&g.updateMatrix(),p.value.copy(g.matrix)}function n(g,p){p.color.getRGB(g.fogColor.value,wc(i)),p.isFog?(g.fogNear.value=p.near,g.fogFar.value=p.far):p.isFogExp2&&(g.fogDensity.value=p.density)}function s(g,p,v,b,_){p.isNodeMaterial?p.uniformsNeedUpdate=!1:p.isMeshBasicMaterial?r(g,p):p.isMeshLambertMaterial?(r(g,p),p.envMap&&(g.envMapIntensity.value=p.envMapIntensity)):p.isMeshToonMaterial?(r(g,p),h(g,p)):p.isMeshPhongMaterial?(r(g,p),u(g,p),p.envMap&&(g.envMapIntensity.value=p.envMapIntensity)):p.isMeshStandardMaterial?(r(g,p),d(g,p),p.isMeshPhysicalMaterial&&f(g,p,_)):p.isMeshMatcapMaterial?(r(g,p),m(g,p)):p.isMeshDepthMaterial?r(g,p):p.isMeshDistanceMaterial?(r(g,p),y(g,p)):p.isMeshNormalMaterial?r(g,p):p.isLineBasicMaterial?(a(g,p),p.isLineDashedMaterial&&o(g,p)):p.isPointsMaterial?c(g,p,v,b):p.isSpriteMaterial?l(g,p):p.isShadowMaterial?(g.color.value.copy(p.color),g.opacity.value=p.opacity):p.isShaderMaterial&&(p.uniformsNeedUpdate=!1)}function r(g,p){g.opacity.value=p.opacity,p.color&&g.diffuse.value.copy(p.color),p.emissive&&g.emissive.value.copy(p.emissive).multiplyScalar(p.emissiveIntensity),p.map&&(g.map.value=p.map,e(p.map,g.mapTransform)),p.alphaMap&&(g.alphaMap.value=p.alphaMap,e(p.alphaMap,g.alphaMapTransform)),p.bumpMap&&(g.bumpMap.value=p.bumpMap,e(p.bumpMap,g.bumpMapTransform),g.bumpScale.value=p.bumpScale,p.side===We&&(g.bumpScale.value*=-1)),p.normalMap&&(g.normalMap.value=p.normalMap,e(p.normalMap,g.normalMapTransform),g.normalScale.value.copy(p.normalScale),p.side===We&&g.normalScale.value.negate()),p.displacementMap&&(g.displacementMap.value=p.displacementMap,e(p.displacementMap,g.displacementMapTransform),g.displacementScale.value=p.displacementScale,g.displacementBias.value=p.displacementBias),p.emissiveMap&&(g.emissiveMap.value=p.emissiveMap,e(p.emissiveMap,g.emissiveMapTransform)),p.specularMap&&(g.specularMap.value=p.specularMap,e(p.specularMap,g.specularMapTransform)),p.alphaTest>0&&(g.alphaTest.value=p.alphaTest);let v=t.get(p),b=v.envMap,_=v.envMapRotation;b&&(g.envMap.value=b,g.envMapRotation.value.setFromMatrix4(c_.makeRotationFromEuler(_)).transpose(),b.isCubeTexture&&b.isRenderTargetTexture===!1&&g.envMapRotation.value.premultiply(_d),g.reflectivity.value=p.reflectivity,g.ior.value=p.ior,g.refractionRatio.value=p.refractionRatio),p.lightMap&&(g.lightMap.value=p.lightMap,g.lightMapIntensity.value=p.lightMapIntensity,e(p.lightMap,g.lightMapTransform)),p.aoMap&&(g.aoMap.value=p.aoMap,g.aoMapIntensity.value=p.aoMapIntensity,e(p.aoMap,g.aoMapTransform))}function a(g,p){g.diffuse.value.copy(p.color),g.opacity.value=p.opacity,p.map&&(g.map.value=p.map,e(p.map,g.mapTransform))}function o(g,p){g.dashSize.value=p.dashSize,g.totalSize.value=p.dashSize+p.gapSize,g.scale.value=p.scale}function c(g,p,v,b){g.diffuse.value.copy(p.color),g.opacity.value=p.opacity,g.size.value=p.size*v,g.scale.value=b*.5,p.map&&(g.map.value=p.map,e(p.map,g.uvTransform)),p.alphaMap&&(g.alphaMap.value=p.alphaMap,e(p.alphaMap,g.alphaMapTransform)),p.alphaTest>0&&(g.alphaTest.value=p.alphaTest)}function l(g,p){g.diffuse.value.copy(p.color),g.opacity.value=p.opacity,g.rotation.value=p.rotation,p.map&&(g.map.value=p.map,e(p.map,g.mapTransform)),p.alphaMap&&(g.alphaMap.value=p.alphaMap,e(p.alphaMap,g.alphaMapTransform)),p.alphaTest>0&&(g.alphaTest.value=p.alphaTest)}function u(g,p){g.specular.value.copy(p.specular),g.shininess.value=Math.max(p.shininess,1e-4)}function h(g,p){p.gradientMap&&(g.gradientMap.value=p.gradientMap)}function d(g,p){g.metalness.value=p.metalness,p.metalnessMap&&(g.metalnessMap.value=p.metalnessMap,e(p.metalnessMap,g.metalnessMapTransform)),g.roughness.value=p.roughness,p.roughnessMap&&(g.roughnessMap.value=p.roughnessMap,e(p.roughnessMap,g.roughnessMapTransform)),p.envMap&&(g.envMapIntensity.value=p.envMapIntensity)}function f(g,p,v){g.ior.value=p.ior,p.diffuseRoughness>0&&(g.diffuseRoughness.value=p.diffuseRoughness,p.diffuseRoughnessMap&&(g.diffuseRoughnessMap.value=p.diffuseRoughnessMap,e(p.diffuseRoughnessMap,g.diffuseRoughnessMapTransform))),p.sheen>0&&(g.sheenColor.value.copy(p.sheenColor).multiplyScalar(p.sheen),g.sheenRoughness.value=p.sheenRoughness,p.sheenColorMap&&(g.sheenColorMap.value=p.sheenColorMap,e(p.sheenColorMap,g.sheenColorMapTransform)),p.sheenRoughnessMap&&(g.sheenRoughnessMap.value=p.sheenRoughnessMap,e(p.sheenRoughnessMap,g.sheenRoughnessMapTransform))),p.clearcoat>0&&(g.clearcoat.value=p.clearcoat,g.clearcoatRoughness.value=p.clearcoatRoughness,p.clearcoatMap&&(g.clearcoatMap.value=p.clearcoatMap,e(p.clearcoatMap,g.clearcoatMapTransform)),p.clearcoatRoughnessMap&&(g.clearcoatRoughnessMap.value=p.clearcoatRoughnessMap,e(p.clearcoatRoughnessMap,g.clearcoatRoughnessMapTransform)),p.clearcoatNormalMap&&(g.clearcoatNormalMap.value=p.clearcoatNormalMap,e(p.clearcoatNormalMap,g.clearcoatNormalMapTransform),g.clearcoatNormalScale.value.copy(p.clearcoatNormalScale),p.side===We&&g.clearcoatNormalScale.value.negate())),p.dispersion>0&&(g.dispersion.value=p.dispersion),p.retroreflectivity>0&&(g.retroreflectivity.value=p.retroreflectivity),p.iridescence>0&&(g.iridescence.value=p.iridescence,g.iridescenceIOR.value=p.iridescenceIOR,g.iridescenceThicknessMinimum.value=p.iridescenceThicknessMinimum,g.iridescenceThicknessMaximum.value=p.iridescenceThicknessMaximum,p.iridescenceMap&&(g.iridescenceMap.value=p.iridescenceMap,e(p.iridescenceMap,g.iridescenceMapTransform)),p.iridescenceThicknessMap&&(g.iridescenceThicknessMap.value=p.iridescenceThicknessMap,e(p.iridescenceThicknessMap,g.iridescenceThicknessMapTransform))),p.transmission>0&&(g.transmission.value=p.transmission,g.transmissionSamplerMap.value=v.texture,g.transmissionSamplerSize.value.set(v.width,v.height),p.transmissionMap&&(g.transmissionMap.value=p.transmissionMap,e(p.transmissionMap,g.transmissionMapTransform)),g.thickness.value=p.thickness,p.thicknessMap&&(g.thicknessMap.value=p.thicknessMap,e(p.thicknessMap,g.thicknessMapTransform)),g.attenuationDistance.value=p.attenuationDistance,g.attenuationColor.value.copy(p.attenuationColor)),p.anisotropy>0&&(g.anisotropyVector.value.set(p.anisotropy*Math.cos(p.anisotropyRotation),p.anisotropy*Math.sin(p.anisotropyRotation)),p.anisotropyMap&&(g.anisotropyMap.value=p.anisotropyMap,e(p.anisotropyMap,g.anisotropyMapTransform))),g.specularIntensity.value=p.specularIntensity,g.specularColor.value.copy(p.specularColor),p.specularColorMap&&(g.specularColorMap.value=p.specularColorMap,e(p.specularColorMap,g.specularColorMapTransform)),p.specularIntensityMap&&(g.specularIntensityMap.value=p.specularIntensityMap,e(p.specularIntensityMap,g.specularIntensityMapTransform))}function m(g,p){p.matcap&&(g.matcap.value=p.matcap)}function y(g,p){let v=t.get(p).light;g.referencePosition.value.setFromMatrixPosition(v.matrixWorld),g.nearDistance.value=v.shadow.camera.near,g.farDistance.value=v.shadow.camera.far}return{refreshFogUniforms:n,refreshMaterialUniforms:s}}function u_(i,t,e,n){let s={},r={};function a(v,b){let _=b.program;n.uniformBlockBinding(v,_,s[v.id])}function o(v,b){let _=s[v.id];_===void 0&&(m(v),_=c(v),s[v.id]=_,v.addEventListener("dispose",g));let T=b.program;n.updateUBOMapping(v,T);let w=t.render.frame;r[v.id]!==w&&(l(v),r[v.id]=w)}function c(v){let b=i.createBuffer(),_=v.__size,T=v.usage;return i.bindBuffer(i.UNIFORM_BUFFER,b),i.bufferData(i.UNIFORM_BUFFER,_,T),i.bindBuffer(i.UNIFORM_BUFFER,null),b}function l(v){let b=s[v.id],_=v.uniforms,T=v.__cache;i.bindBuffer(i.UNIFORM_BUFFER,b);for(let w=0,x=_.length;w<x;w++){let M=_[w];if(Array.isArray(M))for(let P=0,C=M.length;P<C;P++)u(M[P],w,P,T);else u(M,w,0,T)}i.bindBuffer(i.UNIFORM_BUFFER,null)}function u(v,b,_,T){if(d(v,b,_,T)===!0){let w=v.__offset,x=v.value;if(Array.isArray(x)){let M=0;for(let P=0;P<x.length;P++){let C=x[P],k=y(C,!0);h(C,v.__data,M),M+=k.storage/Float32Array.BYTES_PER_ELEMENT}}else h(x,v.__data,0);i.bufferSubData(i.UNIFORM_BUFFER,w,v.__data)}}function h(v,b,_){typeof v=="number"||typeof v=="boolean"?b[_]=v:v.isMatrix3?(b[_]=v.elements[0],b[_+1]=v.elements[1],b[_+2]=v.elements[2],b[_+3]=0,b[_+4]=v.elements[3],b[_+5]=v.elements[4],b[_+6]=v.elements[5],b[_+7]=0,b[_+8]=v.elements[6],b[_+9]=v.elements[7],b[_+10]=v.elements[8],b[_+11]=0):ArrayBuffer.isView(v)?b.set(v.subarray(0,b.length-_),_):v.toArray(b,_)}function d(v,b,_,T){let w=v.value,x=b+"_"+_;if(Array.isArray(w)){T[x]===void 0&&(T[x]=[]);let M=T[x],P=!1;for(let C=0;C<w.length;C++)f(w[C],M,C)===!0&&(P=!0);return P}return f(w,T,x)}function f(v,b,_){if(b[_]===void 0)return typeof v=="number"||typeof v=="boolean"?b[_]=v:ArrayBuffer.isView(v)?b[_]=v.slice():b[_]=v.clone(),!0;{let T=b[_];if(typeof v=="number"||typeof v=="boolean"){if(T!==v)return b[_]=v,!0}else{if(ArrayBuffer.isView(v))return!0;if(T.equals(v)===!1)return T.copy(v),!0}}return!1}function m(v){let b=v.uniforms,_=0,T=16;for(let x=0,M=b.length;x<M;x++){let P=Array.isArray(b[x])?b[x]:[b[x]];for(let C=0,k=P.length;C<k;C++){let O=P[C],j=Array.isArray(O.value)?O.value:[O.value],D=Array.isArray(b[x])||Array.isArray(O.value),K=_;for(let Y=0,J=j.length;Y<J;Y++){let at=j[Y],W=y(at,D),tt=_%T,vt=tt%W.boundary,wt=tt+vt;_+=vt,wt!==0&&T-wt<W.storage&&(_+=T-wt),Y===0&&(K=_),_+=W.storage}let Q=O.type==="int"?Int32Array:O.type==="uint"?Uint32Array:Float32Array;O.__data=new Q((_-K)/Float32Array.BYTES_PER_ELEMENT),O.__offset=K}}let w=_%T;return w>0&&(_+=T-w),v.__size=_,v.__cache={},this}function y(v,b){let _={boundary:0,storage:0};return typeof v=="number"||typeof v=="boolean"?(_.boundary=4,_.storage=4):v.isVector2?(_.boundary=8,_.storage=8):v.isVector3||v.isColor?(_.boundary=16,_.storage=12):v.isVector4?(_.boundary=16,_.storage=16):v.isMatrix3?(_.boundary=16,_.storage=48):v.isMatrix4?(_.boundary=16,_.storage=64):v.isTexture?qt("WebGLRenderer: Texture samplers can not be part of an uniforms group."):ArrayBuffer.isView(v)?(_.boundary=16,_.storage=v.byteLength):qt("WebGLRenderer: Unsupported uniform value type.",v),b===!0&&(_.boundary=16,_.storage=Math.ceil(_.storage/16)*16),_}function g(v){let b=v.target;b.removeEventListener("dispose",g),i.deleteBuffer(s[b.id]),delete s[b.id],delete r[b.id]}function p(){for(let v in s)i.deleteBuffer(s[v]);s={},r={}}return{bind:a,update:o,dispose:p}}var d_=new Uint16Array([12469,15057,12620,14925,13266,14620,13807,14376,14323,13990,14545,13625,14713,13328,14840,12882,14931,12528,14996,12233,15039,11829,15066,11525,15080,11295,15085,10976,15082,10705,15073,10495,13880,14564,13898,14542,13977,14430,14158,14124,14393,13732,14556,13410,14702,12996,14814,12596,14891,12291,14937,11834,14957,11489,14958,11194,14943,10803,14921,10506,14893,10278,14858,9960,14484,14039,14487,14025,14499,13941,14524,13740,14574,13468,14654,13106,14743,12678,14818,12344,14867,11893,14889,11509,14893,11180,14881,10751,14852,10428,14812,10128,14765,9754,14712,9466,14764,13480,14764,13475,14766,13440,14766,13347,14769,13070,14786,12713,14816,12387,14844,11957,14860,11549,14868,11215,14855,10751,14825,10403,14782,10044,14729,9651,14666,9352,14599,9029,14967,12835,14966,12831,14963,12804,14954,12723,14936,12564,14917,12347,14900,11958,14886,11569,14878,11247,14859,10765,14828,10401,14784,10011,14727,9600,14660,9289,14586,8893,14508,8533,15111,12234,15110,12234,15104,12216,15092,12156,15067,12010,15028,11776,14981,11500,14942,11205,14902,10752,14861,10393,14812,9991,14752,9570,14682,9252,14603,8808,14519,8445,14431,8145,15209,11449,15208,11451,15202,11451,15190,11438,15163,11384,15117,11274,15055,10979,14994,10648,14932,10343,14871,9936,14803,9532,14729,9218,14645,8742,14556,8381,14461,8020,14365,7603,15273,10603,15272,10607,15267,10619,15256,10631,15231,10614,15182,10535,15118,10389,15042,10167,14963,9787,14883,9447,14800,9115,14710,8665,14615,8318,14514,7911,14411,7507,14279,7198,15314,9675,15313,9683,15309,9712,15298,9759,15277,9797,15229,9773,15166,9668,15084,9487,14995,9274,14898,8910,14800,8539,14697,8234,14590,7790,14479,7409,14367,7067,14178,6621,15337,8619,15337,8631,15333,8677,15325,8769,15305,8871,15264,8940,15202,8909,15119,8775,15022,8565,14916,8328,14804,8009,14688,7614,14569,7287,14448,6888,14321,6483,14088,6171,15350,7402,15350,7419,15347,7480,15340,7613,15322,7804,15287,7973,15229,8057,15148,8012,15046,7846,14933,7611,14810,7357,14682,7069,14552,6656,14421,6316,14251,5948,14007,5528,15356,5942,15356,5977,15353,6119,15348,6294,15332,6551,15302,6824,15249,7044,15171,7122,15070,7050,14949,6861,14818,6611,14679,6349,14538,6067,14398,5651,14189,5311,13935,4958,15359,4123,15359,4153,15356,4296,15353,4646,15338,5160,15311,5508,15263,5829,15188,6042,15088,6094,14966,6001,14826,5796,14678,5543,14527,5287,14377,4985,14133,4586,13869,4257,15360,1563,15360,1642,15358,2076,15354,2636,15341,3350,15317,4019,15273,4429,15203,4732,15105,4911,14981,4932,14836,4818,14679,4621,14517,4386,14359,4156,14083,3795,13808,3437,15360,122,15360,137,15358,285,15355,636,15344,1274,15322,2177,15281,2765,15215,3223,15120,3451,14995,3569,14846,3567,14681,3466,14511,3305,14344,3121,14037,2800,13753,2467,15360,0,15360,1,15359,21,15355,89,15346,253,15325,479,15287,796,15225,1148,15133,1492,15008,1749,14856,1882,14685,1886,14506,1783,14324,1608,13996,1398,13702,1183]),kn=null;function f_(){return kn===null&&(kn=new hr(d_,16,16,Ti,Ln),kn.name="DFG_LUT",kn.minFilter=je,kn.magFilter=je,kn.wrapS=mn,kn.wrapT=mn,kn.generateMipmaps=!1,kn.needsUpdate=!0),kn}var Gn=class{constructor(t={}){let{canvas:e=Eu(),context:n=null,depth:s=!0,stencil:r=!1,alpha:a=!1,antialias:o=!1,premultipliedAlpha:c=!0,preserveDrawingBuffer:l=!1,powerPreference:u="default",failIfMajorPerformanceCaveat:h=!1,reversedDepthBuffer:d=!1,outputBufferType:f=nn}=t;this.isWebGLRenderer=!0;let m;if(n!==null){if(typeof WebGLRenderingContext<"u"&&n instanceof WebGLRenderingContext)throw new Error("THREE.WebGLRenderer: WebGL 1 is not supported since r163.");m=n.getContextAttributes().alpha}else m=a;let y=f,g=new Set([go,mo,po]),p=new Set([nn,In,Us,$i,ho,uo]),v=new Uint32Array(4),b=new Int32Array(4),_={textures:[],min:[],max:[],resolution:[]},T=null,w=null,x=[],M=[],P=null;this.domElement=e,this.debug={checkShaderErrors:!0,diagnostics:{keywords:!1},onShaderError:null},this.autoClear=!0,this.autoClearColor=!0,this.autoClearDepth=!0,this.autoClearStencil=!0,this.sortObjects=!0,this.clippingPlanes=[],this.localClippingEnabled=!1,this.toneMapping=Bn,this.toneMappingExposure=1,this.transmissionResolutionScale=1;let C=this,k=!1,O=null,j=null,D=null,K=null;this._outputColorSpace=He;let Q=0,Y=0,J=null,at=-1,W=null,tt=new Se,vt=new Se,wt=null,st=new Vt(0),U=0,et=e.width,V=e.height,N=1,L=null,I=null,G=new Se(0,0,et,V),ht=new Se(0,0,et,V),ut=!1,E=new _i,$=new za,B=!1,nt=!1,it=new xe,pt=new F,mt=new Se,X={background:null,fog:null,environment:null,overrideMaterial:null,isScene:!0},Z=!1;function bt(){return J===null?N:1}let H=n;function Qt(A,q){return e.getContext(A,q)}let jt,se,Mt,fe,Gt,$t,R,S,rt,St,yt,Et,Rt,gt,xt,Pt,z,_t,dt,It,At,Ct,Wt;try{let A={alpha:!0,depth:s,stencil:r,antialias:o,premultipliedAlpha:c,preserveDrawingBuffer:l,powerPreference:u,failIfMajorPerformanceCaveat:h};if("setAttribute"in e&&e.setAttribute("data-engine",`three.js r${"187"}`),e.addEventListener("webglcontextlost",Yt,!1),e.addEventListener("webglcontextrestored",te,!1),e.addEventListener("webglcontextcreationerror",Nt,!1),H===null){let q="webgl2";if(H=Qt(q,A),H===null)throw Qt(q)?new Error("THREE.WebGLRenderer: Error creating WebGL context with your selected attributes."):new Error("THREE.WebGLRenderer: Error creating WebGL context.")}ee()}catch(A){throw e.removeEventListener("webglcontextlost",Yt,!1),e.removeEventListener("webglcontextrestored",te,!1),e.removeEventListener("webglcontextcreationerror",Nt,!1),Zt("WebGLRenderer: "+A.message),A}function ee(){jt=new w0(H),jt.init(),At=new a_(H,jt),se=new f0(H,jt,t,At),Mt=new s_(H,jt),se.reversedDepthBuffer&&d&&Mt.buffers.depth.setReversed(!0),j=H.createFramebuffer(),D=H.createFramebuffer(),K=H.createFramebuffer(),fe=new E0(H),Gt=new Wx,$t=new r_(H,jt,Mt,Gt,se,At,fe),R=new S0(C),S=new Rp(H),Ct=new u0(H,S),rt=new T0(H,S,fe,Ct),St=new R0(H,rt,S,Ct,fe),_t=new C0(H,se,$t),xt=new p0(Gt),yt=new Hx(C,R,jt,se,Ct,xt),Et=new h_(C,Gt),Rt=new qx,gt=new jx(jt),z=new h0(C,R,Mt,St,m,c),Pt=new i_(C,St,se),Wt=new u_(H,fe,se,Mt),dt=new d0(H,jt,fe),It=new A0(H,jt,fe),fe.programs=yt.programs,C.capabilities=se,C.extensions=jt,C.properties=Gt,C.renderLists=Rt,C.shadowMap=Pt,C.state=Mt,C.info=fe}y!==nn&&(P=new I0(y,e.width,e.height,o,s,r));let Tt=new Oc(C,H);this.xr=Tt,this.getContext=function(){return H},this.getContextAttributes=function(){return H.getContextAttributes()},this.forceContextLoss=function(){let A=jt.get("WEBGL_lose_context");A&&A.loseContext()},this.forceContextRestore=function(){let A=jt.get("WEBGL_lose_context");A&&A.restoreContext()},this.getPixelRatio=function(){return N},this.setPixelRatio=function(A){A!==void 0&&(N=A,this.setSize(et,V,!1))},this.getSize=function(A){return A.set(et,V)},this.setSize=function(A,q,ct=!0){if(Tt.isPresenting){qt("WebGLRenderer: Can't change size while VR device is presenting.");return}et=A,V=q,e.width=Math.floor(A*N),e.height=Math.floor(q*N),ct===!0&&(e.style.width=A+"px",e.style.height=q+"px"),P!==null&&P.setSize(e.width,e.height),this.setViewport(0,0,A,q)},this.getDrawingBufferSize=function(A){return A.set(et*N,V*N).floor()},this.setDrawingBufferSize=function(A,q,ct){et=A,V=q,N=ct,e.width=Math.floor(A*ct),e.height=Math.floor(q*ct),this.setViewport(0,0,A,q)},this.setEffects=function(A){if(y===nn){Zt("WebGLRenderer: setEffects() requires outputBufferType set to HalfFloatType or FloatType.");return}if(A){for(let q=0;q<A.length;q++)if(A[q].isOutputPass===!0){qt("WebGLRenderer: OutputPass is not needed in setEffects(). Tone mapping and color space conversion are applied automatically.");break}}P.setEffects(A||[])},this.getCurrentViewport=function(A){return A.copy(tt)},this.getViewport=function(A){return A.copy(G)},this.setViewport=function(A,q,ct,ot){A.isVector4?G.set(A.x,A.y,A.z,A.w):G.set(A,q,ct,ot),Mt.viewport(tt.copy(G).multiplyScalar(bt()).round())},this.getScissor=function(A){return A.copy(ht)},this.setScissor=function(A,q,ct,ot){A.isVector4?ht.set(A.x,A.y,A.z,A.w):ht.set(A,q,ct,ot),Mt.scissor(vt.copy(ht).multiplyScalar(bt()).round())},this.getScissorTest=function(){return ut},this.setScissorTest=function(A){Mt.setScissorTest(ut=A)},this.setOpaqueSort=function(A){L=A},this.setTransparentSort=function(A){I=A},this.getClearColor=function(A){return A.copy(z.getClearColor())},this.setClearColor=function(){z.setClearColor(...arguments)},this.getClearAlpha=function(){return z.getClearAlpha()},this.setClearAlpha=function(){z.setClearAlpha(...arguments)},this.clear=function(A=!0,q=!0,ct=!0){let ot=0;if(A){let lt=!1;if(J!==null){let Dt=J.texture.format;lt=g.has(Dt)}if(lt){let Dt=J.texture.type,Ot=p.has(Dt),Lt=z.getClearColor(),Ft=z.getClearAlpha(),zt=Lt.r,Kt=Lt.g,re=Lt.b;Ot?(v[0]=zt,v[1]=Kt,v[2]=re,v[3]=Ft,H.clearBufferuiv(H.COLOR,0,v)):(b[0]=zt,b[1]=Kt,b[2]=re,b[3]=Ft,H.clearBufferiv(H.COLOR,0,b))}else ot|=H.COLOR_BUFFER_BIT}q&&(ot|=H.DEPTH_BUFFER_BIT,this.state.buffers.depth.setMask(!0)),ct&&(ot|=H.STENCIL_BUFFER_BIT,this.state.buffers.stencil.setMask(4294967295)),ot!==0&&H.clear(ot)},this.clearColor=function(){this.clear(!0,!1,!1)},this.clearDepth=function(){this.clear(!1,!0,!1)},this.clearStencil=function(){this.clear(!1,!1,!0)},this.setNodesHandler=function(A){A.setRenderer(this),O=A},this.dispose=function(){e.removeEventListener("webglcontextlost",Yt,!1),e.removeEventListener("webglcontextrestored",te,!1),e.removeEventListener("webglcontextcreationerror",Nt,!1),z.dispose(),Rt.dispose(),gt.dispose(),$t.dispose(),Gt.dispose(),R.dispose(),St.dispose(),rt.dispose(),Ct.dispose(),Wt.dispose(),yt.dispose(),Tt.dispose(),Tt.removeEventListener("sessionstart",Qc),Tt.removeEventListener("sessionend",th),Ri.stop()};function Yt(A){A.preventDefault(),yc("WebGLRenderer: Context Lost."),k=!0}function te(){yc("WebGLRenderer: Context Restored."),k=!1;let A=fe.autoReset,q=Pt.enabled,ct=Pt.autoUpdate,ot=Pt.needsUpdate,lt=Pt.type;ee(),fe.autoReset=A,Pt.enabled=q,Pt.autoUpdate=ct,Pt.needsUpdate=ot,Pt.type=lt}function Nt(A){Zt("WebGLRenderer: A WebGL context could not be created. Reason: ",A.statusMessage)}function _e(A){let q=A.target;q.removeEventListener("dispose",_e),Be(q)}function Be(A){ci(A),Gt.remove(A)}function ci(A){let q=Gt.get(A).programs;q!==void 0&&(q.forEach(function(ct){yt.releaseProgram(ct)}),A.isShaderMaterial&&yt.releaseShaderCache(A))}this.renderBufferDirect=function(A,q,ct,ot,lt,Dt){q===null&&(q=X);let Ot=lt.isMesh&&lt.matrixWorld.determinantAffine()<0,Lt=nf(A,q,ct,ot,lt);Mt.setMaterial(ot,Ot);let Ft=ct.index,zt=1;if(ot.wireframe===!0){if(Ft=rt.getWireframeAttribute(ct),Ft===void 0)return;zt=2}let Kt=ct.drawRange,re=ct.attributes.position,Bt=Kt.start*zt,ae=(Kt.start+Kt.count)*zt;Dt!==null&&(Bt=Math.max(Bt,Dt.start*zt),ae=Math.min(ae,(Dt.start+Dt.count)*zt)),Ft!==null?(Bt=Math.max(Bt,0),ae=Math.min(ae,Ft.count)):re!=null&&(Bt=Math.max(Bt,0),ae=Math.min(ae,re.count));let Pe=ae-Bt;if(Pe<0||Pe===1/0)return;Ct.setup(lt,ot,Lt,ct,Ft);let be,ve=dt;if(Ft!==null&&(be=S.get(Ft),ve=It,ve.setIndex(be)),lt.isMesh)ot.wireframe===!0?(Mt.setLineWidth(ot.wireframeLinewidth*bt()),ve.setMode(H.LINES)):ve.setMode(H.TRIANGLES);else if(lt.isLine){let Xe=ot.linewidth;Xe===void 0&&(Xe=1),Mt.setLineWidth(Xe*bt()),lt.isLineSegments?ve.setMode(H.LINES):lt.isLineLoop?ve.setMode(H.LINE_LOOP):ve.setMode(H.LINE_STRIP)}else lt.isPoints?ve.setMode(H.POINTS):lt.isSprite&&ve.setMode(H.TRIANGLES);if(lt.isBatchedMesh)if(jt.get("WEBGL_multi_draw"))ve.renderMultiDraw(lt._multiDrawStarts,lt._multiDrawCounts,lt._multiDrawCount);else{let Xe=lt._multiDrawStarts,kt=lt._multiDrawCounts,Je=lt._multiDrawCount,pe=Ft?S.get(Ft).bytesPerElement:1,Pi=Gt.get(ot).currentProgram.getUniforms();for(let qe=0;qe<Je;qe++)Pi.setValue(H,"_gl_DrawID",qe),ve.render(Xe[qe]/pe,kt[qe])}else if(lt.isInstancedMesh)ve.renderInstances(Bt,Pe,lt.count);else if(ct.isInstancedBufferGeometry){let Xe=ct._maxInstanceCount!==void 0?ct._maxInstanceCount:1/0,kt=Math.min(ct.instanceCount,Xe);ve.renderInstances(Bt,Pe,kt)}else ve.render(Bt,Pe)};function jc(A,q,ct,ot){O!==null&&A.isNodeMaterial&&O.setObject(ot,A),B===!0&&xt.setState(A,ct,!1),A.transparent===!0&&A.side===Ee&&A.forceSinglePass===!1?(A.side=We,A.needsUpdate=!0,Gs(A,q,ot),A.side=wi,A.needsUpdate=!0,Gs(A,q,ot),A.side=Ee):Gs(A,q,ot)}this.compile=function(A,q,ct=null){ct===null&&(ct=A),O!==null&&O.renderStart(A,q,ct),w=gt.get(ct),w.init(q),M.push(w),ct.traverseVisible(function(lt){lt.isLight&&lt.layers.test(q.layers)&&(w.pushLight(lt),lt.castShadow&&w.pushShadow(lt))}),A!==ct&&A.traverseVisible(function(lt){lt.isLight&&lt.layers.test(q.layers)&&(w.pushLight(lt),lt.castShadow&&w.pushShadow(lt))}),w.setupLights(),O!==null&&O.updateLights(w.state.lightsArray),nt=this.localClippingEnabled,B=xt.init(this.clippingPlanes,nt),B===!0&&xt.setGlobalState(this.clippingPlanes,q),O!==null&&Pt.render(w.state.shadowsArray,ct,q);let ot=new Set;return A.traverse(function(lt){if(!(lt.isMesh||lt.isPoints||lt.isLine||lt.isSprite))return;let Dt=lt.material;if(Dt)if(Array.isArray(Dt))for(let Ot=0;Ot<Dt.length;Ot++){let Lt=Dt[Ot];jc(Lt,ct,q,lt),ot.add(Lt)}else jc(Dt,ct,q,lt),ot.add(Dt)}),w=M.pop(),O!==null&&O.renderEnd(),ot},this.compileAsync=function(A,q,ct=null){let ot=this.compile(A,q,ct);return new Promise(lt=>{function Dt(){if(ot.forEach(function(Ot){let Ft=Gt.get(Ot).currentProgram;(Ft===void 0||Ft.isReady())&&ot.delete(Ot)}),ot.size===0){lt(A);return}setTimeout(Dt,10)}jt.get("KHR_parallel_shader_compile")!==null?Dt():setTimeout(Dt,10)})};let cl=null;function tf(A){cl&&cl(A)}function Qc(){Ri.stop()}function th(){Ri.start()}let Ri=new ld;Ri.setAnimationLoop(tf),typeof self<"u"&&Ri.setContext(self),this.setAnimationLoop=function(A){cl=A,Tt.setAnimationLoop(A),A===null?Ri.stop():Ri.start()},Tt.addEventListener("sessionstart",Qc),Tt.addEventListener("sessionend",th),this.render=function(A,q){if(q!==void 0&&q.isCamera!==!0){Zt("WebGLRenderer.render: camera is not an instance of THREE.Camera.");return}if(k===!0)return;O!==null&&O.renderStart(A,q);let ct=Tt.enabled===!0&&Tt.isPresenting===!0,ot=P!==null&&(J===null||ct)&&P.begin(C,J);if(A.matrixWorldAutoUpdate===!0&&A.updateMatrixWorld(),q.parent===null&&q.matrixWorldAutoUpdate===!0&&q.updateMatrixWorld(),Tt.enabled===!0&&Tt.isPresenting===!0&&(P===null||P.isCompositing()===!1)&&(Tt.cameraAutoUpdate===!0&&Tt.updateCamera(q),q=Tt.getCamera()),A.isScene===!0&&A.onBeforeRender(C,A,q,J),w=gt.get(A,M.length),w.init(q),w.state.textureUnits=$t.getTextureUnits(),M.push(w),it.multiplyMatrices(q.projectionMatrix,q.matrixWorldInverse),q.isArrayCamera?$.setFromArrayCamera(q):E.setFromProjectionMatrix(it,gn,q.reversedDepth),nt=this.localClippingEnabled,B=xt.init(this.clippingPlanes,nt),T=Rt.get(A,x.length),T.init(),x.push(T),this.info.render.frame++,Tt.enabled===!0&&Tt.isPresenting===!0){let Ot=C.xr.getDepthSensingMesh();Ot!==null&&hl(Ot,q,-1/0,C.sortObjects)}hl(A,q,0,C.sortObjects),T.finish(),O!==null&&O.updateLights(w.state.lightsArray),C.sortObjects===!0&&T.sort(L,I),Z=Tt.enabled===!1||Tt.isPresenting===!1||Tt.hasDepthSensing()===!1,Z&&z.addToRenderList(T,A),this.info.autoReset===!0&&this.info.reset(),B===!0&&xt.beginShadows();let lt=w.state.shadowsArray;if(Pt.render(lt,A,q),B===!0&&xt.endShadows(),(ot&&P.hasRenderPass())===!1){let Ot=T.opaque,Lt=T.transmissive;if(w.setupLights(),q.isArrayCamera){let Ft=q.cameras;if(Lt.length>0)for(let zt=0,Kt=Ft.length;zt<Kt;zt++){let re=Ft[zt];nh(Ot,Lt,A,re)}Z&&z.render(A);for(let zt=0,Kt=Ft.length;zt<Kt;zt++){let re=Ft[zt];eh(T,A,re,re.viewport)}}else Lt.length>0&&nh(Ot,Lt,A,q),Z&&z.render(A),eh(T,A,q)}J!==null&&Y===0&&($t.updateMultisampleRenderTarget(J),$t.updateRenderTargetMipmap(J)),ot&&P.end(C),A.isScene===!0&&A.onAfterRender(C,A,q),Ct.resetDefaultState(),at=-1,W=null,M.pop(),M.length>0?(w=M[M.length-1],$t.setTextureUnits(w.state.textureUnits),B===!0&&xt.setGlobalState(C.clippingPlanes,w.state.camera)):w=null,x.pop(),x.length>0?T=x[x.length-1]:T=null,O!==null&&O.renderEnd()};function hl(A,q,ct,ot){if(A.visible===!1)return;if(A.layers.test(q.layers)){if(A.isGroup)ct=A.renderOrder;else if(A.isLOD)A.autoUpdate===!0&&A.update(q);else if(A.isLight)w.pushLight(A),A.castShadow&&w.pushShadow(A);else if(A.isSprite){let Ot=q.isArrayCamera?$:E;if(!A.frustumCulled||A.intersectsFrustum(Ot)){ot&&mt.setFromMatrixPosition(A.matrixWorld).applyMatrix4(it);let Lt=St.update(A),Ft=A.material;Ft.visible&&T.push(A,Lt,Ft,ct,mt.z,null,q)}}else if(A.isMesh||A.isLine||A.isPoints){let Ot=q.isArrayCamera?$:E;if(!A.frustumCulled||A.intersectsFrustum(Ot)){let Lt=St.update(A),Ft=A.material;if(ot&&(A.boundingSphere!==void 0?(A.boundingSphere===null&&A.computeBoundingSphere(),mt.copy(A.boundingSphere.center)):(Lt.boundingSphere===null&&Lt.computeBoundingSphere(),mt.copy(Lt.boundingSphere.center)),mt.applyMatrix4(A.matrixWorld).applyMatrix4(it)),Array.isArray(Ft)){let zt=Lt.groups;for(let Kt=0,re=zt.length;Kt<re;Kt++){let Bt=zt[Kt],ae=Ft[Bt.materialIndex];ae&&ae.visible&&T.push(A,Lt,ae,ct,mt.z,Bt,q)}}else Ft.visible&&T.push(A,Lt,Ft,ct,mt.z,null,q)}}}let Dt=A.children;for(let Ot=0,Lt=Dt.length;Ot<Lt;Ot++)hl(Dt[Ot],q,ct,ot)}function eh(A,q,ct,ot){let{opaque:lt,transmissive:Dt,transparent:Ot}=A;w.setupLightsView(ct),B===!0&&xt.setGlobalState(C.clippingPlanes,ct),ot&&Mt.viewport(tt.copy(ot)),lt.length>0&&ta(lt,q,ct),Dt.length>0&&ta(Dt,q,ct),Ot.length>0&&ta(Ot,q,ct),Mt.buffers.depth.setTest(!0),Mt.buffers.depth.setMask(!0),Mt.buffers.color.setMask(!0),Mt.setPolygonOffset(!1)}function nh(A,q,ct,ot){if((ct.isScene===!0?ct.overrideMaterial:null)!==null)return;if(w.state.transmissionRenderTarget[ot.id]===void 0){let Bt=jt.has("EXT_color_buffer_half_float")||jt.has("EXT_color_buffer_float");w.state.transmissionRenderTarget[ot.id]=new cn(1,1,{generateMipmaps:!0,type:Bt?Ln:nn,minFilter:si,samples:Math.max(4,se.samples),stencilBuffer:r,resolveDepthBuffer:!1,resolveStencilBuffer:!1,storeMultisampledDepthBuffer:!1,storeMultisampledStencilBuffer:!1,colorSpace:oe.workingColorSpace})}let Dt=w.state.transmissionRenderTarget[ot.id],Ot=ot.viewport||tt;Dt.setSize(Ot.z*C.transmissionResolutionScale,Ot.w*C.transmissionResolutionScale);let Lt=C.getRenderTarget(),Ft=C.getActiveCubeFace(),zt=C.getActiveMipmapLevel();C.setRenderTarget(Dt),C.getClearColor(st),U=C.getClearAlpha(),U<1&&C.setClearColor(16777215,.5),C.clear(),Z&&z.render(ct);let Kt=C.toneMapping;C.toneMapping=Bn;let re=ot.viewport;if(ot.viewport!==void 0&&(ot.viewport=void 0),w.setupLightsView(ot),B===!0&&xt.setGlobalState(C.clippingPlanes,ot),ta(A,ct,ot),$t.updateMultisampleRenderTarget(Dt),$t.updateRenderTargetMipmap(Dt),jt.has("WEBGL_multisampled_render_to_texture")===!1){let Bt=!1;for(let ae=0,Pe=q.length;ae<Pe;ae++){let be=q[ae],{object:ve,geometry:Xe,material:kt,group:Je}=be;if(kt.side===Ee&&ve.layers.test(ot.layers)){let pe=kt.side;kt.side=We,kt.needsUpdate=!0,ih(ve,ct,ot,Xe,kt,Je),kt.side=pe,kt.needsUpdate=!0,Bt=!0}}Bt===!0&&($t.updateMultisampleRenderTarget(Dt),$t.updateRenderTargetMipmap(Dt))}C.setRenderTarget(Lt,Ft,zt),C.setClearColor(st,U),re!==void 0&&(ot.viewport=re),C.toneMapping=Kt}function ta(A,q,ct){let ot=q.isScene===!0?q.overrideMaterial:null;for(let lt=0,Dt=A.length;lt<Dt;lt++){let Ot=A[lt],{object:Lt,geometry:Ft,group:zt}=Ot,Kt=Ot.material;Kt.allowOverride===!0&&ot!==null&&(Kt=ot),Lt.layers.test(ct.layers)&&ih(Lt,q,ct,Ft,Kt,zt)}}function ih(A,q,ct,ot,lt,Dt){O!==null&&lt.isNodeMaterial&&O.setObject(A,lt),A.onBeforeRender(C,q,ct,ot,lt,Dt),A.modelViewMatrix.multiplyMatrices(ct.matrixWorldInverse,A.matrixWorld),A.normalMatrix.getNormalMatrix(A.modelViewMatrix),lt.onBeforeRender(C,q,ct,ot,A,Dt),lt.transparent===!0&&lt.side===Ee&&lt.forceSinglePass===!1?(lt.side=We,lt.needsUpdate=!0,C.renderBufferDirect(ct,q,ot,lt,A,Dt),lt.side=wi,lt.needsUpdate=!0,C.renderBufferDirect(ct,q,ot,lt,A,Dt),lt.side=Ee):C.renderBufferDirect(ct,q,ot,lt,A,Dt),A.onAfterRender(C,q,ct,ot,lt,Dt)}function sh(A){let q=0;for(let ct of A)ct.texture!==null&&q++;return q}function Gs(A,q,ct){q.isScene!==!0&&(q=X);let ot=Gt.get(A),lt=w.state.lights,Dt=w.state.shadowsArray,Ot=lt.state.version,Lt=yt.getParameters(A,lt.state,Dt,q,ct,w.state.lightProbeGridArray),Ft=yt.getProgramCacheKey(Lt),zt=ot.programs;ot.environment=A.isMeshStandardMaterial||A.isMeshLambertMaterial||A.isMeshPhongMaterial?q.environment:null,ot.fog=q.fog;let Kt=A.isMeshStandardMaterial||A.isMeshLambertMaterial&&!A.envMap||A.isMeshPhongMaterial&&!A.envMap;ot.envMap=R.get(A.envMap||ot.environment,Kt),ot.envMapRotation=ot.environment!==null&&A.envMap===null?q.environmentRotation:A.envMapRotation,zt===void 0&&(A.addEventListener("dispose",_e),zt=new Map,ot.programs=zt,ot.programVariants=new Map);let re=zt.get(Ft);if(re!==void 0){if(ot.currentProgram===re&&ot.lightsStateVersion===Ot)return ah(A,Lt),re}else Lt.uniforms=yt.getUniforms(A),O!==null&&A.isNodeMaterial&&O.build(A,ct,Lt),A.onBeforeCompile(Lt,C),re=yt.acquireProgram(Lt,Ft),zt.set(Ft,re),ot.uniforms=Lt.uniforms;let Bt=ot.uniforms;return(!A.isShaderMaterial&&!A.isRawShaderMaterial||A.clipping===!0)&&(Bt.clippingPlanes=xt.uniform),ah(A,Lt),ot.needsLights=rf(A),ot.lightsStateVersion=Ot,ot.needsLights&&(Bt.ambientLightColor.value=lt.state.ambient,Bt.lightProbe.value=lt.state.probe,Bt.sunLights.value=lt.state.sun,Bt.sunLightShadows.value=lt.state.sunShadow,Bt.directionalLights.value=lt.state.directional,Bt.directionalLightShadows.value=lt.state.directionalShadow,Bt.spotLights.value=lt.state.spot,Bt.spotLightShadows.value=lt.state.spotShadow,Bt.rectAreaLights.value=lt.state.rectArea,Bt.ltc_1.value=lt.state.rectAreaLTC1,Bt.ltc_2.value=lt.state.rectAreaLTC2,Bt.pointLights.value=lt.state.point,Bt.pointLightShadows.value=lt.state.pointShadow,Bt.hemisphereLights.value=lt.state.hemi,Bt.sunShadowMatrix.value=lt.state.sunShadowMatrix,Bt.sunShadowCascade.value=lt.state.sunShadowCascade,Bt.directionalShadowMatrix.value=lt.state.directionalShadowMatrix,Bt.spotLightMatrix.value=lt.state.spotLightMatrix,Bt.spotLightMap.value=lt.state.spotLightMap,Bt.pointShadowMatrix.value=lt.state.pointShadowMatrix),ot.numLightProbeGrids=sh(w.state.lightProbeGridArray),ot.currentProgram=re,ot.uniformsList=null,re}function ef(A,q,ct,ot,lt){let Dt=Gt.get(A),Ot=Dt.currentProgram.id+","+ot+","+lt,Lt=Dt.programVariants.get(Ot);return Lt===void 0?(Lt=Gs(A,q,ct),Dt.programVariants.set(Ot,Lt)):(Dt.currentProgram=Lt,Dt.uniformsList=null,Dt.outputColorSpace=ot,Dt.toneMapping=lt),Lt}function rh(A){if(A.uniformsList===null){let q=A.currentProgram.getUniforms();A.uniformsList=Fs.seqWithValue(q.seq,A.uniforms)}return A.uniformsList}function ah(A,q){let ct=Gt.get(A);ct.outputColorSpace=q.outputColorSpace,ct.batching=q.batching,ct.batchingColor=q.batchingColor,ct.instancing=q.instancing,ct.instancingColor=q.instancingColor,ct.instancingMorph=q.instancingMorph,ct.skinning=q.skinning,ct.morphTargets=q.morphTargets,ct.morphNormals=q.morphNormals,ct.morphColors=q.morphColors,ct.morphTargetsCount=q.morphTargetsCount,ct.numClippingPlanes=q.numClippingPlanes,ct.numIntersection=q.numClipIntersection,ct.vertexAlphas=q.vertexAlphas,ct.vertexTangents=q.vertexTangents,ct.toneMapping=q.toneMapping}function nf(A,q,ct,ot,lt){q.isScene!==!0&&(q=X),$t.resetTextureUnits();let Dt=q.fog,Ot=ot.isMeshStandardMaterial||ot.isMeshLambertMaterial||ot.isMeshPhongMaterial?q.environment:null,Lt=J===null?C.outputColorSpace:J.isXRRenderTarget===!0?J.texture.colorSpace:oe.workingColorSpace,Ft=ot.isMeshStandardMaterial||ot.isMeshLambertMaterial&&!ot.envMap||ot.isMeshPhongMaterial&&!ot.envMap,zt=R.get(ot.envMap||Ot,Ft),Kt=ot.vertexColors===!0&&!!ct.attributes.color&&ct.attributes.color.itemSize===4,re=!!ct.attributes.tangent&&(!!ot.normalMap||ot.anisotropy>0),Bt=!!ct.morphAttributes.position,ae=!!ct.morphAttributes.normal,Pe=!!ct.morphAttributes.color,be=Bn;ot.toneMapped&&(J===null||J.isXRRenderTarget===!0)&&(be=C.toneMapping);let ve=ct.morphAttributes.position||ct.morphAttributes.normal||ct.morphAttributes.color,Xe=ve!==void 0?ve.length:0,kt=Gt.get(ot),Je=w.state.lights;if(B===!0&&(nt===!0||A!==W)){let we=A===W&&ot.id===at;xt.setState(ot,A,we)}let pe=!1,Pi=!1;ot.version===kt.__version?kt.needsLights&&kt.lightsStateVersion!==Je.state.version||lt.isBatchedMesh&&kt.batching===!1||!lt.isBatchedMesh&&kt.batching===!0||lt.isBatchedMesh&&kt.batchingColor===!0&&lt._colorsTexture===null||lt.isBatchedMesh&&kt.batchingColor===!1&&lt._colorsTexture!==null||lt.isInstancedMesh&&kt.instancing===!1||!lt.isInstancedMesh&&kt.instancing===!0||lt.isSkinnedMesh&&kt.skinning===!1||!lt.isSkinnedMesh&&kt.skinning===!0||lt.isInstancedMesh&&kt.instancingColor===!0&&lt.instanceColor===null||lt.isInstancedMesh&&kt.instancingColor===!1&&lt.instanceColor!==null||lt.isInstancedMesh&&kt.instancingMorph===!0&&lt.morphTexture===null||lt.isInstancedMesh&&kt.instancingMorph===!1&&lt.morphTexture!==null||kt.envMap!==zt||ot.fog===!0&&kt.fog!==Dt||kt.numClippingPlanes!==void 0&&(kt.numClippingPlanes!==xt.numPlanes||kt.numIntersection!==xt.numIntersection)||kt.vertexAlphas!==Kt||kt.vertexTangents!==re||kt.morphTargets!==Bt||kt.morphNormals!==ae||kt.morphColors!==Pe||kt.morphTargetsCount!==Xe||kt.numLightProbeGrids!==sh(w.state.lightProbeGridArray)?pe=!0:(kt.outputColorSpace!==Lt||kt.toneMapping!==be)&&(Pi=!0):(pe=!0,kt.__version=ot.version);let qe=kt.currentProgram;(pe===!0||Pi===!0)&&(qe=Pi===!0?ef(ot,q,lt,Lt,be):Gs(ot,q,lt),O&&ot.isNodeMaterial&&O.onUpdateProgram(ot,qe,kt));let ea=!1,hi=!1,Ii=!1,le=qe.getUniforms(),Ve=kt.uniforms;if(Mt.useProgram(qe.program)&&(ea=!0,hi=!0,Ii=!0),ot.id!==at&&(at=ot.id,hi=!0),ea||W!==A){Mt.buffers.depth.getReversed()&&A.reversedDepth!==!0&&(A._reversedDepth=!0,A.updateProjectionMatrix()),le.setValue(H,"projectionMatrix",A.projectionMatrix),le.setValue(H,"viewMatrix",A.matrixWorldInverse);let ke=le.map.cameraPosition;ke!==void 0&&ke.setValue(H,pt.setFromMatrixPosition(A.matrixWorld)),se.logarithmicDepthBuffer&&le.setValue(H,"logDepthBufFC",2/(Math.log(A.far+1)/Math.LN2)),(ot.isMeshPhongMaterial||ot.isMeshToonMaterial||ot.isMeshLambertMaterial||ot.isMeshBasicMaterial||ot.isMeshStandardMaterial||ot.isShaderMaterial)&&le.setValue(H,"isOrthographic",A.isOrthographicCamera===!0),W!==A&&(W=A,hi=!0,Ii=!0)}if(kt.needsLights&&(Je.state.sunShadowMap.length>0&&le.setValue(H,"sunShadowMap",Je.state.sunShadowMap,$t),Je.state.directionalShadowMap.length>0&&le.setValue(H,"directionalShadowMap",Je.state.directionalShadowMap,$t),Je.state.spotShadowMap.length>0&&le.setValue(H,"spotShadowMap",Je.state.spotShadowMap,$t),Je.state.pointShadowMap.length>0&&le.setValue(H,"pointShadowMap",Je.state.pointShadowMap,$t)),lt.isSkinnedMesh){le.setOptional(H,lt,"bindMatrix"),le.setOptional(H,lt,"bindMatrixInverse");let we=lt.skeleton;we&&(we.boneTexture===null&&we.computeBoneTexture(),le.setValue(H,"boneTexture",we.boneTexture,$t))}lt.isBatchedMesh&&(le.setOptional(H,lt,"batchingTexture"),le.setValue(H,"batchingTexture",lt._matricesTexture,$t),le.setOptional(H,lt,"batchingIdTexture"),le.setValue(H,"batchingIdTexture",lt._indirectTexture,$t),le.setOptional(H,lt,"batchingColorTexture"),lt._colorsTexture!==null&&le.setValue(H,"batchingColorTexture",lt._colorsTexture,$t));let _n=ct.morphAttributes;if((_n.position!==void 0||_n.normal!==void 0||_n.color!==void 0)&&_t.update(lt,ct,qe),(hi||kt.receiveShadow!==lt.receiveShadow)&&(kt.receiveShadow=lt.receiveShadow,le.setValue(H,"receiveShadow",lt.receiveShadow)),(ot.isMeshStandardMaterial||ot.isMeshLambertMaterial||ot.isMeshPhongMaterial)&&ot.envMap===null&&q.environment!==null&&(Ve.envMapIntensity.value=q.environmentIntensity),Ve.dfgLUT!==void 0&&(Ve.dfgLUT.value=f_()),hi){if(le.setValue(H,"toneMappingExposure",C.toneMappingExposure),kt.needsLights&&sf(Ve,Ii),Dt&&ot.fog===!0&&Et.refreshFogUniforms(Ve,Dt),Et.refreshMaterialUniforms(Ve,ot,N,V,w.state.transmissionRenderTarget[A.id]),kt.needsLights&&kt.numLightProbeGrids>0){let we=_,ke=0;for(let Hn of w.state.lightProbeGridArray)Hn.texture!==null&&(we.textures[ke]=Hn.texture,we.min[ke]=Hn.boundingBox.min,we.max[ke]=Hn.boundingBox.max,we.resolution[ke]=Hn.resolution,ke++);for(let Hn in we)we[Hn].length=ke;Ve.probesSH.value=we.textures,Ve.probesMin.value=we.min,Ve.probesMax.value=we.max,Ve.probesResolution.value=we.resolution}Fs.upload(H,rh(kt),Ve,$t)}if(ot.isShaderMaterial&&ot.uniformsNeedUpdate===!0&&(Fs.upload(H,rh(kt),Ve,$t),ot.uniformsNeedUpdate=!1),ot.isSpriteMaterial&&le.setValue(H,"center",lt.center),le.setValue(H,"modelViewMatrix",lt.modelViewMatrix),le.setValue(H,"normalMatrix",lt.normalMatrix),le.setValue(H,"modelMatrix",lt.matrixWorld),ot.uniformsGroups!==void 0){let we=ot.uniformsGroups;for(let ke=0,Hn=we.length;ke<Hn;ke++){let lh=we[ke];Wt.update(lh,qe),Wt.bind(lh,qe)}}return qe}function sf(A,q){A.ambientLightColor.needsUpdate=q,A.lightProbe.needsUpdate=q,A.sunLights.needsUpdate=q,A.sunLightShadows.needsUpdate=q,A.directionalLights.needsUpdate=q,A.directionalLightShadows.needsUpdate=q,A.pointLights.needsUpdate=q,A.pointLightShadows.needsUpdate=q,A.spotLights.needsUpdate=q,A.spotLightShadows.needsUpdate=q,A.rectAreaLights.needsUpdate=q,A.hemisphereLights.needsUpdate=q}function rf(A){return A.isMeshLambertMaterial||A.isMeshToonMaterial||A.isMeshPhongMaterial||A.isMeshStandardMaterial||A.isShadowMaterial||A.isShaderMaterial&&A.lights===!0}this.getActiveCubeFace=function(){return Q},this.getActiveMipmapLevel=function(){return Y},this.getRenderTarget=function(){return J},this.setRenderTargetTextures=function(A,q,ct){let ot=Gt.get(A);ot.__autoAllocateDepthBuffer=A.resolveDepthBuffer===!1,ot.__autoAllocateDepthBuffer===!1&&(ot.__useRenderToTexture=!1),Gt.get(A.texture).__webglTexture=q,Gt.get(A.depthTexture).__webglTexture=ot.__autoAllocateDepthBuffer?void 0:ct,ot.__hasExternalTextures=!0},this.setRenderTargetFramebuffer=function(A,q){let ct=Gt.get(A);ct.__webglFramebuffer=q,ct.__useDefaultFramebuffer=q===void 0},this.setRenderTarget=function(A,q=0,ct=0){J=A,Q=q,Y=ct;let ot=null,lt=!1,Dt=!1;if(A){let Lt=Gt.get(A);if(Lt.__useDefaultFramebuffer!==void 0){Mt.bindFramebuffer(H.FRAMEBUFFER,Lt.__webglFramebuffer),tt.copy(A.viewport),vt.copy(A.scissor),wt=A.scissorTest,Mt.viewport(tt),Mt.scissor(vt),Mt.setScissorTest(wt),at=-1;return}else if(Lt.__webglFramebuffer===void 0)$t.setupRenderTarget(A);else if(Lt.__hasExternalTextures)$t.rebindTextures(A,Gt.get(A.texture).__webglTexture,Gt.get(A.depthTexture).__webglTexture);else if(A.depthBuffer){let Kt=A.depthTexture;if(Lt.__boundDepthTexture!==Kt){if(Kt!==null&&Gt.has(Kt)&&(A.width!==Kt.image.width||A.height!==Kt.image.height))throw new Error("THREE.WebGLRenderer: Attached DepthTexture is initialized to the incorrect size.");$t.setupDepthRenderbuffer(A)}}let Ft=A.texture;(Ft.isData3DTexture||Ft.isDataArrayTexture||Ft.isCompressedArrayTexture)&&(Dt=!0);let zt=Gt.get(A).__webglFramebuffer;A.isWebGLCubeRenderTarget?(Array.isArray(zt[q])?ot=zt[q][ct]:ot=zt[q],lt=!0):A.samples>0&&$t.useMultisampledRTT(A)===!1?ot=Gt.get(A).__webglMultisampledFramebuffer:Array.isArray(zt)?ot=zt[ct]:ot=zt,tt.copy(A.viewport),vt.copy(A.scissor),wt=A.scissorTest}else tt.copy(G).multiplyScalar(N).floor(),vt.copy(ht).multiplyScalar(N).floor(),wt=ut;if(ct!==0&&(ot=j),Mt.bindFramebuffer(H.FRAMEBUFFER,ot)&&Mt.drawBuffers(A,ot),Mt.viewport(tt),Mt.scissor(vt),Mt.setScissorTest(wt),lt){let Lt=Gt.get(A.texture);H.framebufferTexture2D(H.FRAMEBUFFER,H.COLOR_ATTACHMENT0,H.TEXTURE_CUBE_MAP_POSITIVE_X+q,Lt.__webglTexture,ct)}else if(Dt){let Lt=q;for(let Ft=0;Ft<A.textures.length;Ft++){let zt=Gt.get(A.textures[Ft]);H.framebufferTextureLayer(H.FRAMEBUFFER,H.COLOR_ATTACHMENT0+Ft,zt.__webglTexture,ct,Lt)}}else if(A!==null&&ct!==0){let Lt=Gt.get(A.texture);H.framebufferTexture2D(H.FRAMEBUFFER,H.COLOR_ATTACHMENT0,H.TEXTURE_2D,Lt.__webglTexture,ct)}at=-1};function oh(A){let q=Gt.get(A);return(q.__readFormat!==A.format||q.__readType!==A.type)&&(q.__readFormat=A.format,q.__readType=A.type,q.__formatReadable=se.textureFormatReadable(A.format),q.__typeReadable=se.textureTypeReadable(A.type)),q}this.readRenderTargetPixels=function(A,q,ct,ot,lt,Dt,Ot,Lt=0){if(!(A&&A.isWebGLRenderTarget)){Zt("WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");return}let Ft=Gt.get(A).__webglFramebuffer;if(A.isWebGLCubeRenderTarget&&Ot!==void 0&&(Ft=Ft[Ot],Array.isArray(Ft)&&(Ft=Ft[0])),Ft){Mt.bindFramebuffer(H.FRAMEBUFFER,Ft);try{let zt=A.textures[Lt],Kt=zt.format,re=zt.type;A.textures.length>1&&H.readBuffer(H.COLOR_ATTACHMENT0+Lt);let Bt=oh(zt);if(Bt.__formatReadable===!1){Zt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in RGBA or implementation defined format.");return}if(Bt.__typeReadable===!1){Zt("WebGLRenderer.readRenderTargetPixels: renderTarget is not in UnsignedByteType or implementation defined type.");return}q>=0&&q<=A.width-ot&&ct>=0&&ct<=A.height-lt&&H.readPixels(q,ct,ot,lt,At.convert(Kt),At.convert(re),Dt)}finally{let zt=J!==null?Gt.get(J).__webglFramebuffer:null;Mt.bindFramebuffer(H.FRAMEBUFFER,zt)}}},this.readRenderTargetPixelsAsync=async function(A,q,ct,ot,lt,Dt,Ot,Lt=0){if(!(A&&A.isWebGLRenderTarget))throw new Error("THREE.WebGLRenderer.readRenderTargetPixels: renderTarget is not THREE.WebGLRenderTarget.");let Ft=Gt.get(A).__webglFramebuffer;if(A.isWebGLCubeRenderTarget&&Ot!==void 0&&(Ft=Ft[Ot],Array.isArray(Ft)&&(Ft=Ft[0])),Ft)if(q>=0&&q<=A.width-ot&&ct>=0&&ct<=A.height-lt){Mt.bindFramebuffer(H.FRAMEBUFFER,Ft);let zt=A.textures[Lt],Kt=zt.format,re=zt.type;A.textures.length>1&&H.readBuffer(H.COLOR_ATTACHMENT0+Lt);let Bt=oh(zt);if(Bt.__formatReadable===!1)throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in RGBA or implementation defined format.");if(Bt.__typeReadable===!1)throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: renderTarget is not in UnsignedByteType or implementation defined type.");let ae=H.createBuffer();H.bindBuffer(H.PIXEL_PACK_BUFFER,ae),H.bufferData(H.PIXEL_PACK_BUFFER,Dt.byteLength,H.STREAM_READ),H.readPixels(q,ct,ot,lt,At.convert(Kt),At.convert(re),0),H.bindBuffer(H.PIXEL_PACK_BUFFER,null);let Pe=J!==null?Gt.get(J).__webglFramebuffer:null;Mt.bindFramebuffer(H.FRAMEBUFFER,Pe);let be=H.fenceSync(H.SYNC_GPU_COMMANDS_COMPLETE,0);return H.flush(),await Ru(H,be,4),H.bindBuffer(H.PIXEL_PACK_BUFFER,ae),H.getBufferSubData(H.PIXEL_PACK_BUFFER,0,Dt),H.bindBuffer(H.PIXEL_PACK_BUFFER,null),H.deleteBuffer(ae),H.deleteSync(be),Dt}else throw new Error("THREE.WebGLRenderer.readRenderTargetPixelsAsync: requested read bounds are out of range.")},this.copyFramebufferToTexture=function(A,q=null,ct=0){let ot=Math.pow(2,-ct),lt=Math.floor(A.image.width*ot),Dt=Math.floor(A.image.height*ot),Ot=q!==null?q.x:0,Lt=q!==null?q.y:0;$t.setTexture2D(A,0),H.copyTexSubImage2D(H.TEXTURE_2D,ct,0,0,Ot,Lt,lt,Dt),Mt.unbindTexture()},this.copyTextureToTexture=function(A,q,ct=null,ot=null,lt=0,Dt=0){let Ot,Lt,Ft,zt,Kt,re,Bt,ae,Pe,be=A.isCompressedTexture?A.mipmaps[Dt]:A.image;if(ct!==null)Ot=ct.max.x-ct.min.x,Lt=ct.max.y-ct.min.y,Ft=ct.isBox3?ct.max.z-ct.min.z:1,zt=ct.min.x,Kt=ct.min.y,re=ct.isBox3?ct.min.z:0;else{let le=Math.pow(2,-lt);Ot=Math.floor(be.width*le),Lt=Math.floor(be.height*le),A.isDataArrayTexture?Ft=be.depth:A.isData3DTexture?Ft=Math.floor(be.depth*le):Ft=1,zt=0,Kt=0,re=0}ot!==null?(Bt=ot.x,ae=ot.y,Pe=ot.z):(Bt=0,ae=0,Pe=0);let ve=At.convert(q.format),Xe=At.convert(q.type),kt;q.isData3DTexture?($t.setTexture3D(q,0),kt=H.TEXTURE_3D):q.isDataArrayTexture||q.isCompressedArrayTexture?($t.setTexture2DArray(q,0),kt=H.TEXTURE_2D_ARRAY):($t.setTexture2D(q,0),kt=H.TEXTURE_2D),Mt.activeTexture(H.TEXTURE0),Mt.pixelStorei(H.UNPACK_FLIP_Y_WEBGL,q.flipY),Mt.pixelStorei(H.UNPACK_PREMULTIPLY_ALPHA_WEBGL,q.premultiplyAlpha),Mt.pixelStorei(H.UNPACK_ALIGNMENT,q.unpackAlignment);let Je=Mt.getParameter(H.UNPACK_ROW_LENGTH),pe=Mt.getParameter(H.UNPACK_IMAGE_HEIGHT),Pi=Mt.getParameter(H.UNPACK_SKIP_PIXELS),qe=Mt.getParameter(H.UNPACK_SKIP_ROWS),ea=Mt.getParameter(H.UNPACK_SKIP_IMAGES);Mt.pixelStorei(H.UNPACK_ROW_LENGTH,be.width),Mt.pixelStorei(H.UNPACK_IMAGE_HEIGHT,be.height),Mt.pixelStorei(H.UNPACK_SKIP_PIXELS,zt),Mt.pixelStorei(H.UNPACK_SKIP_ROWS,Kt),Mt.pixelStorei(H.UNPACK_SKIP_IMAGES,re);let hi=A.isDataArrayTexture||A.isData3DTexture,Ii=q.isDataArrayTexture||q.isData3DTexture;if(A.isDepthTexture){let le=Gt.get(A),Ve=Gt.get(q),_n=Gt.get(le.__renderTarget),we=Gt.get(Ve.__renderTarget);Mt.bindFramebuffer(H.READ_FRAMEBUFFER,_n.__webglFramebuffer),Mt.bindFramebuffer(H.DRAW_FRAMEBUFFER,we.__webglFramebuffer);for(let ke=0;ke<Ft;ke++)hi&&(H.framebufferTextureLayer(H.READ_FRAMEBUFFER,H.COLOR_ATTACHMENT0,Gt.get(A).__webglTexture,lt,re+ke),H.framebufferTextureLayer(H.DRAW_FRAMEBUFFER,H.COLOR_ATTACHMENT0,Gt.get(q).__webglTexture,Dt,Pe+ke)),H.blitFramebuffer(zt,Kt,zt+Ot,Kt+Lt,Bt,ae,Bt+Ot,ae+Lt,H.DEPTH_BUFFER_BIT,H.NEAREST);Mt.bindFramebuffer(H.READ_FRAMEBUFFER,null),Mt.bindFramebuffer(H.DRAW_FRAMEBUFFER,null)}else if(lt!==0||A.isRenderTargetTexture||Gt.has(A)){let le=Gt.get(A),Ve=Gt.get(q);Mt.bindFramebuffer(H.READ_FRAMEBUFFER,D),Mt.bindFramebuffer(H.DRAW_FRAMEBUFFER,K);for(let _n=0;_n<Ft;_n++)hi?H.framebufferTextureLayer(H.READ_FRAMEBUFFER,H.COLOR_ATTACHMENT0,le.__webglTexture,lt,re+_n):H.framebufferTexture2D(H.READ_FRAMEBUFFER,H.COLOR_ATTACHMENT0,H.TEXTURE_2D,le.__webglTexture,lt),Ii?H.framebufferTextureLayer(H.DRAW_FRAMEBUFFER,H.COLOR_ATTACHMENT0,Ve.__webglTexture,Dt,Pe+_n):H.framebufferTexture2D(H.DRAW_FRAMEBUFFER,H.COLOR_ATTACHMENT0,H.TEXTURE_2D,Ve.__webglTexture,Dt),lt!==0?H.blitFramebuffer(zt,Kt,zt+Ot,Kt+Lt,Bt,ae,Bt+Ot,ae+Lt,H.COLOR_BUFFER_BIT,H.NEAREST):Ii?H.copyTexSubImage3D(kt,Dt,Bt,ae,Pe+_n,zt,Kt,Ot,Lt):H.copyTexSubImage2D(kt,Dt,Bt,ae,zt,Kt,Ot,Lt);Mt.bindFramebuffer(H.READ_FRAMEBUFFER,null),Mt.bindFramebuffer(H.DRAW_FRAMEBUFFER,null)}else Ii?A.isDataTexture||A.isData3DTexture?H.texSubImage3D(kt,Dt,Bt,ae,Pe,Ot,Lt,Ft,ve,Xe,be.data):q.isCompressedArrayTexture?H.compressedTexSubImage3D(kt,Dt,Bt,ae,Pe,Ot,Lt,Ft,ve,be.data):H.texSubImage3D(kt,Dt,Bt,ae,Pe,Ot,Lt,Ft,ve,Xe,be):A.isDataTexture?H.texSubImage2D(H.TEXTURE_2D,Dt,Bt,ae,Ot,Lt,ve,Xe,be.data):A.isCompressedTexture?H.compressedTexSubImage2D(H.TEXTURE_2D,Dt,Bt,ae,be.width,be.height,ve,be.data):H.texSubImage2D(H.TEXTURE_2D,Dt,Bt,ae,Ot,Lt,ve,Xe,be);Mt.pixelStorei(H.UNPACK_ROW_LENGTH,Je),Mt.pixelStorei(H.UNPACK_IMAGE_HEIGHT,pe),Mt.pixelStorei(H.UNPACK_SKIP_PIXELS,Pi),Mt.pixelStorei(H.UNPACK_SKIP_ROWS,qe),Mt.pixelStorei(H.UNPACK_SKIP_IMAGES,ea),Dt===0&&q.generateMipmaps===!0&&q.mipmapsAutoUpdate===!0&&H.generateMipmap(kt),Mt.unbindTexture()},this.initRenderTarget=function(A){Gt.get(A).__webglFramebuffer===void 0&&$t.setupRenderTarget(A)},this.initTexture=function(A){A.isCubeTexture?$t.setTextureCube(A,0):A.isData3DTexture?$t.setTexture3D(A,0):A.isDataArrayTexture||A.isCompressedArrayTexture?$t.setTexture2DArray(A,0):$t.setTexture2D(A,0),Mt.unbindTexture()},this.resetState=function(){Q=0,Y=0,J=null,Mt.reset(),Ct.reset()},typeof __THREE_DEVTOOLS__<"u"&&__THREE_DEVTOOLS__.dispatchEvent(new CustomEvent("observe",{detail:this}))}get coordinateSystem(){return gn}get outputColorSpace(){return this._outputColorSpace}set outputColorSpace(t){this._outputColorSpace=t;let e=this.getContext();e.drawingBufferColorSpace=oe._getDrawingBufferColorSpace(t),e.unpackColorSpace=oe._getUnpackColorSpace()}};var Xr=He,qr=class i extends Wi{constructor(t){super(t),this.defaultDPI=90,this.defaultUnit="px"}load(t,e,n,s){let r=this,a=new Rr(r.manager);a.setPriority(r.priority),a.setCache(r.cache),a.setPath(r.path),a.setRequestHeader(r.requestHeader),a.setWithCredentials(r.withCredentials),a.load(t,function(o){try{e(r.parse(o))}catch(c){s?s(c):console.error(c),r.manager.itemError(t)}},n,s)}parse(t){let e=this;function n(N,L){if(N.nodeType!==1)return;N.hasAttribute("filter")&&console.warn("THREE.SVGLoader: Filters are not supported.");let I=T(N),G=!1,ht=null;switch(N.nodeName){case"svg":L=y(N,L);break;case"style":r(N);break;case"g":L=y(N,L);break;case"path":L=y(N,L),N.hasAttribute("d")&&(ht=s(N));break;case"rect":L=y(N,L),ht=c(N);break;case"polygon":L=y(N,L),ht=l(N);break;case"polyline":L=y(N,L),ht=u(N);break;case"circle":L=y(N,L),ht=h(N);break;case"ellipse":L=y(N,L),ht=d(N);break;case"line":L=y(N,L),ht=f(N);break;case"defs":G=!0;break;case"use":L=y(N,L);let $=(N.getAttributeNS("http://www.w3.org/1999/xlink","href")||N.getAttribute("href")||"").substring(1),B=N.viewportElement.getElementById($);B?n(B,L):console.warn("SVGLoader: 'use node' references non-existent node id: "+$);break;default:}if(ht){L.fill!==void 0&&L.fill!=="none"&&!L.fill.startsWith("url")&&ht.color.setStyle(L.fill,Xr),M(ht,U),K.push(ht);let E=Object.assign({},L);E.strokeWidth=L.strokeWidth*j(U),ht.userData={node:N,style:E,transform:U.clone(),gradients:Y}}let ut=N.childNodes;for(let E=0;E<ut.length;E++){let $=ut[E];G&&$.nodeName!=="style"&&$.nodeName!=="defs"||n($,L)}I&&(J.pop(),J.length>0?U.copy(J[J.length-1]):U.identity())}function s(N){let L=new Rn,I=new ft,G=new ft,ht=new ft,ut=!1,E=N.getAttribute("d");if(E===""||E==="none")return null;let $=E.match(/[a-df-z][^a-df-z]*/ig);for(let B=0,nt=$.length;B<nt;B++){let it=$[B],pt=it.charAt(0),mt=it.slice(1).trim();ut===!0&&(pt!=="M"&&pt!=="m"&&pt!=="Z"&&pt!=="z"&&L.moveTo(ht.x,ht.y),ut=!1);let X;switch(pt){case"M":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z+=2)I.x=X[Z+0],I.y=X[Z+1],G.x=I.x,G.y=I.y,Z===0?L.moveTo(I.x,I.y):L.lineTo(I.x,I.y),Z===0&&ht.copy(I);break;case"H":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z++)I.x=X[Z],G.x=I.x,G.y=I.y,L.lineTo(I.x,I.y);break;case"V":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z++)I.y=X[Z],G.x=I.x,G.y=I.y,L.lineTo(I.x,I.y);break;case"L":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z+=2)I.x=X[Z+0],I.y=X[Z+1],G.x=I.x,G.y=I.y,L.lineTo(I.x,I.y);break;case"C":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z+=6)L.bezierCurveTo(X[Z+0],X[Z+1],X[Z+2],X[Z+3],X[Z+4],X[Z+5]),G.x=X[Z+2],G.y=X[Z+3],I.x=X[Z+4],I.y=X[Z+5];break;case"S":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z+=4)L.bezierCurveTo(g(I.x,G.x),g(I.y,G.y),X[Z+0],X[Z+1],X[Z+2],X[Z+3]),G.x=X[Z+0],G.y=X[Z+1],I.x=X[Z+2],I.y=X[Z+3];break;case"Q":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z+=4)L.quadraticCurveTo(X[Z+0],X[Z+1],X[Z+2],X[Z+3]),G.x=X[Z+0],G.y=X[Z+1],I.x=X[Z+2],I.y=X[Z+3];break;case"T":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z+=2){let H=g(I.x,G.x),Qt=g(I.y,G.y);L.quadraticCurveTo(H,Qt,X[Z+0],X[Z+1]),G.x=H,G.y=Qt,I.x=X[Z+0],I.y=X[Z+1]}break;case"A":X=p(mt,[3,4],7);for(let Z=0,bt=X.length;Z<bt;Z+=7){if(X[Z+5]==I.x&&X[Z+6]==I.y)continue;let H=I.clone();I.x=X[Z+5],I.y=X[Z+6],G.x=I.x,G.y=I.y,a(L,X[Z],X[Z+1],X[Z+2],X[Z+3],X[Z+4],H,I)}break;case"m":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z+=2)I.x+=X[Z+0],I.y+=X[Z+1],G.x=I.x,G.y=I.y,Z===0?L.moveTo(I.x,I.y):L.lineTo(I.x,I.y),Z===0&&ht.copy(I);break;case"h":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z++)I.x+=X[Z],G.x=I.x,G.y=I.y,L.lineTo(I.x,I.y);break;case"v":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z++)I.y+=X[Z],G.x=I.x,G.y=I.y,L.lineTo(I.x,I.y);break;case"l":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z+=2)I.x+=X[Z+0],I.y+=X[Z+1],G.x=I.x,G.y=I.y,L.lineTo(I.x,I.y);break;case"c":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z+=6)L.bezierCurveTo(I.x+X[Z+0],I.y+X[Z+1],I.x+X[Z+2],I.y+X[Z+3],I.x+X[Z+4],I.y+X[Z+5]),G.x=I.x+X[Z+2],G.y=I.y+X[Z+3],I.x+=X[Z+4],I.y+=X[Z+5];break;case"s":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z+=4)L.bezierCurveTo(g(I.x,G.x),g(I.y,G.y),I.x+X[Z+0],I.y+X[Z+1],I.x+X[Z+2],I.y+X[Z+3]),G.x=I.x+X[Z+0],G.y=I.y+X[Z+1],I.x+=X[Z+2],I.y+=X[Z+3];break;case"q":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z+=4)L.quadraticCurveTo(I.x+X[Z+0],I.y+X[Z+1],I.x+X[Z+2],I.y+X[Z+3]),G.x=I.x+X[Z+0],G.y=I.y+X[Z+1],I.x+=X[Z+2],I.y+=X[Z+3];break;case"t":X=p(mt);for(let Z=0,bt=X.length;Z<bt;Z+=2){let H=g(I.x,G.x),Qt=g(I.y,G.y);L.quadraticCurveTo(H,Qt,I.x+X[Z+0],I.y+X[Z+1]),G.x=H,G.y=Qt,I.x=I.x+X[Z+0],I.y=I.y+X[Z+1]}break;case"a":X=p(mt,[3,4],7);for(let Z=0,bt=X.length;Z<bt;Z+=7){if(X[Z+5]==0&&X[Z+6]==0)continue;let H=I.clone();I.x+=X[Z+5],I.y+=X[Z+6],G.x=I.x,G.y=I.y,a(L,X[Z],X[Z+1],X[Z+2],X[Z+3],X[Z+4],H,I)}break;case"Z":case"z":L.currentPath.autoClose=!0,L.currentPath.curves.length>0&&(I.copy(ht),L.currentPath.currentPoint.copy(I),ut=!0);break;default:console.warn(it)}}return L}function r(N){if(!(!N.sheet||!N.sheet.cssRules||!N.sheet.cssRules.length))for(let L=0;L<N.sheet.cssRules.length;L++){let I=N.sheet.cssRules[L];if(I.type!==1)continue;let G=I.selectorText.split(/,/gm).filter(Boolean).map(ut=>ut.trim()),ht={};for(let ut=0;ut<I.style.length;ut++){let E=I.style[ut];ht[E]=I.style.getPropertyValue(E)}for(let ut=0;ut<G.length;ut++)Q[G[ut]]=Object.assign(Q[G[ut]]||{},ht)}}function a(N,L,I,G,ht,ut,E,$){if(L==0||I==0){N.lineTo($.x,$.y);return}G=G*Math.PI/180,L=Math.abs(L),I=Math.abs(I);let B=(E.x-$.x)/2,nt=(E.y-$.y)/2,it=Math.cos(G)*B+Math.sin(G)*nt,pt=-Math.sin(G)*B+Math.cos(G)*nt,mt=L*L,X=I*I,Z=it*it,bt=pt*pt,H=Z/mt+bt/X;if(H>1){let rt=Math.sqrt(H);L=rt*L,I=rt*I,mt=L*L,X=I*I}let Qt=mt*bt+X*Z,jt=(mt*X-Qt)/Qt,se=Math.sqrt(Math.max(0,jt));ht===ut&&(se=-se);let Mt=se*L*pt/I,fe=-se*I*it/L,Gt=Math.cos(G)*Mt-Math.sin(G)*fe+(E.x+$.x)/2,$t=Math.sin(G)*Mt+Math.cos(G)*fe+(E.y+$.y)/2,R=o(1,0,(it-Mt)/L,(pt-fe)/I),S=o((it-Mt)/L,(pt-fe)/I,(-it-Mt)/L,(-pt-fe)/I)%(Math.PI*2);N.currentPath.absellipse(Gt,$t,L,I,R,R+S,ut===0,G)}function o(N,L,I,G){let ht=N*I+L*G,ut=Math.sqrt(N*N+L*L)*Math.sqrt(I*I+G*G),E=Math.acos(Math.max(-1,Math.min(1,ht/ut)));return N*G-L*I<0&&(E=-E),E}function c(N){let L=_(N.getAttribute("x")||0),I=_(N.getAttribute("y")||0),G=_(N.getAttribute("rx")||N.getAttribute("ry")||0),ht=_(N.getAttribute("ry")||N.getAttribute("rx")||0),ut=_(N.getAttribute("width")),E=_(N.getAttribute("height")),$=1-.551915024494,B=new Rn;return B.moveTo(L+G,I),B.lineTo(L+ut-G,I),(G!==0||ht!==0)&&B.bezierCurveTo(L+ut-G*$,I,L+ut,I+ht*$,L+ut,I+ht),B.lineTo(L+ut,I+E-ht),(G!==0||ht!==0)&&B.bezierCurveTo(L+ut,I+E-ht*$,L+ut-G*$,I+E,L+ut-G,I+E),B.lineTo(L+G,I+E),(G!==0||ht!==0)&&B.bezierCurveTo(L+G*$,I+E,L,I+E-ht*$,L,I+E-ht),B.lineTo(L,I+ht),(G!==0||ht!==0)&&B.bezierCurveTo(L,I+ht*$,L+G*$,I,L+G,I),B}function l(N){function L(ut,E,$){let B=_(E),nt=_($);ht===0?G.moveTo(B,nt):G.lineTo(B,nt),ht++}let I=/([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?)(?:,|\s)([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?)/g,G=new Rn,ht=0;return N.getAttribute("points").replace(I,L),G.currentPath.autoClose=!0,G}function u(N){function L(ut,E,$){let B=_(E),nt=_($);ht===0?G.moveTo(B,nt):G.lineTo(B,nt),ht++}let I=/([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?)(?:,|\s)([+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:e[+-]?\d+)?)/g,G=new Rn,ht=0;return N.getAttribute("points").replace(I,L),G.currentPath.autoClose=!1,G}function h(N){let L=_(N.getAttribute("cx")||0),I=_(N.getAttribute("cy")||0),G=_(N.getAttribute("r")||0),ht=new On;ht.absarc(L,I,G,0,Math.PI*2);let ut=new Rn;return ut.subPaths.push(ht),ut}function d(N){let L=_(N.getAttribute("cx")||0),I=_(N.getAttribute("cy")||0),G=_(N.getAttribute("rx")||0),ht=_(N.getAttribute("ry")||0),ut=new On;ut.absellipse(L,I,G,ht,0,Math.PI*2);let E=new Rn;return E.subPaths.push(ut),E}function f(N){let L=_(N.getAttribute("x1")||0),I=_(N.getAttribute("y1")||0),G=_(N.getAttribute("x2")||0),ht=_(N.getAttribute("y2")||0),ut=new Rn;return ut.moveTo(L,I),ut.lineTo(G,ht),ut.currentPath.autoClose=!1,ut}function m(N){let L="http://www.w3.org/1999/xlink",I=N.querySelectorAll("linearGradient, radialGradient"),G=["x1","y1","x2","y2","cx","cy","r","fx","fy","gradientUnits","gradientTransform","spreadMethod"],ht={};for(let E of I){let $=E.getAttribute("id");if(!$)continue;let B={type:E.nodeName==="radialGradient"?"radialGradient":"linearGradient",attrs:{},stops:null,href:null},nt=E.getAttributeNS(L,"href")||E.getAttribute("href")||"";nt.startsWith("#")&&(B.href=nt.substring(1));for(let pt of G)E.hasAttribute(pt)&&(B.attrs[pt]=E.getAttribute(pt));let it=E.querySelectorAll("stop");if(it.length>0){B.stops=[];for(let pt of it){let mt=pt.getAttribute("stop-color");!mt&&pt.style&&(mt=pt.style["stop-color"]),mt||(mt="#000");let X=pt.getAttribute("stop-opacity");(X===null||X==="")&&pt.style&&(X=pt.style["stop-opacity"]),X=X===null||X===""||X===void 0?1:Math.max(0,Math.min(1,parseFloat(X)));let Z=Math.max(0,Math.min(1,parseFloat(pt.getAttribute("offset")||"0")));B.stops.push({offset:Z,color:mt,opacity:X})}}ht[$]=B}function ut(E,$){let B=ht[E];if(!B||$.has(E))return B;if($.add(E),B.href&&ht[B.href]){let nt=ut(B.href,$);if(nt){B.stops||(B.stops=nt.stops);for(let it in nt.attrs)it in B.attrs||(B.attrs[it]=nt.attrs[it])}}return B}for(let E in ht)ut(E,new Set);for(let E in ht){let pt=function(mt){return typeof mt!="string"?0:mt.endsWith("%")?parseFloat(mt)/100:_(mt)},$=ht[E],B=$.attrs,nt=B.gradientUnits==="userSpaceOnUse"?"userSpaceOnUse":"objectBoundingBox",it={type:$.type,gradientUnits:nt,spreadMethod:B.spreadMethod==="reflect"||B.spreadMethod==="repeat"?B.spreadMethod:"pad",gradientTransform:null,stops:($.stops||[]).slice().sort((mt,X)=>mt.offset-X.offset)};if(B.gradientTransform&&(it.gradientTransform=new Xt,x(B.gradientTransform,it.gradientTransform)),$.type==="linearGradient")it.x1=B.x1!==void 0?pt(B.x1):0,it.y1=B.y1!==void 0?pt(B.y1):0,it.x2=B.x2!==void 0?pt(B.x2):nt==="objectBoundingBox"?1:0,it.y2=B.y2!==void 0?pt(B.y2):0;else{let mt=nt==="objectBoundingBox"?.5:0,X=nt==="objectBoundingBox"?.5:0;it.cx=B.cx!==void 0?pt(B.cx):mt,it.cy=B.cy!==void 0?pt(B.cy):mt,it.r=B.r!==void 0?pt(B.r):X,it.fx=B.fx!==void 0?pt(B.fx):it.cx,it.fy=B.fy!==void 0?pt(B.fy):it.cy}Y[E]=it}}function y(N,L){L=Object.assign({},L);let I={};if(N.hasAttribute("class")){let E=N.getAttribute("class").split(/\s/).filter(Boolean).map($=>$.trim());for(let $=0;$<E.length;$++)I=Object.assign(I,Q["."+E[$]])}N.hasAttribute("id")&&(I=Object.assign(I,Q["#"+N.getAttribute("id")]));function G(E,$,B){B===void 0&&(B=function(it){return it}),N.hasAttribute(E)&&(L[$]=B(N.getAttribute(E))),I[E]&&(L[$]=B(I[E])),N.style&&N.style[E]!==""&&(L[$]=B(N.style[E]))}function ht(E){return Math.max(0,Math.min(1,_(E)))}function ut(E){return Math.max(0,_(E))}return G("fill","fill"),G("fill-opacity","fillOpacity",ht),G("fill-rule","fillRule"),G("opacity","opacity",ht),G("stroke","stroke"),G("stroke-opacity","strokeOpacity",ht),G("stroke-width","strokeWidth",ut),G("stroke-linejoin","strokeLineJoin"),G("stroke-linecap","strokeLineCap"),G("stroke-miterlimit","strokeMiterLimit",ut),G("visibility","visibility"),L}function g(N,L){return N-(L-N)}function p(N,L,I){if(typeof N!="string")throw new TypeError("Invalid input: "+typeof N);let G={SEPARATOR:/[ \t\r\n\,.\-+]/,WHITESPACE:/[ \t\r\n]/,DIGIT:/[\d]/,SIGN:/[-+]/,POINT:/\./,COMMA:/,/,EXP:/e/i,FLAGS:/[01]/},ht=0,ut=1,E=2,$=3,B=ht,nt=!0,it="",pt="",mt=[];function X(Qt,jt,se){let Mt=new SyntaxError('Unexpected character "'+Qt+'" at index '+jt+".");throw Mt.partial=se,Mt}function Z(){it!==""&&(pt===""?mt.push(Number(it)):mt.push(Number(it)*Math.pow(10,Number(pt)))),it="",pt=""}let bt,H=N.length;for(let Qt=0;Qt<H;Qt++){if(bt=N[Qt],Array.isArray(L)&&L.includes(mt.length%I)&&G.FLAGS.test(bt)){B=ut,it=bt,Z();continue}if(B===ht){if(G.WHITESPACE.test(bt))continue;if(G.DIGIT.test(bt)||G.SIGN.test(bt)){B=ut,it=bt;continue}if(G.POINT.test(bt)){B=E,it=bt;continue}G.COMMA.test(bt)&&(nt&&X(bt,Qt,mt),nt=!0)}if(B===ut){if(G.DIGIT.test(bt)){it+=bt;continue}if(G.POINT.test(bt)){it+=bt,B=E;continue}if(G.EXP.test(bt)){B=$;continue}G.SIGN.test(bt)&&it.length===1&&G.SIGN.test(it[0])&&X(bt,Qt,mt)}if(B===E){if(G.DIGIT.test(bt)){it+=bt;continue}if(G.EXP.test(bt)){B=$;continue}G.POINT.test(bt)&&it[it.length-1]==="."&&X(bt,Qt,mt)}if(B===$){if(G.DIGIT.test(bt)){pt+=bt;continue}if(G.SIGN.test(bt)){if(pt===""){pt+=bt;continue}pt.length===1&&G.SIGN.test(pt)&&X(bt,Qt,mt)}}G.WHITESPACE.test(bt)?(Z(),B=ht,nt=!1):G.COMMA.test(bt)?(Z(),B=ht,nt=!0):G.SIGN.test(bt)?(Z(),B=ut,it=bt):G.POINT.test(bt)?(Z(),B=E,it=bt):X(bt,Qt,mt)}return Z(),mt}let v=["mm","cm","in","pt","pc","px"],b={mm:{mm:1,cm:.1,in:1/25.4,pt:72/25.4,pc:6/25.4,px:-1},cm:{mm:10,cm:1,in:1/2.54,pt:72/2.54,pc:6/2.54,px:-1},in:{mm:25.4,cm:2.54,in:1,pt:72,pc:6,px:-1},pt:{mm:25.4/72,cm:2.54/72,in:1/72,pt:1,pc:6/72,px:-1},pc:{mm:25.4/6,cm:2.54/6,in:1/6,pt:72/6,pc:1,px:-1},px:{px:1}};function _(N){let L="px";if(typeof N=="string"||N instanceof String)for(let G=0,ht=v.length;G<ht;G++){let ut=v[G];if(N.endsWith(ut)){L=ut,N=N.substring(0,N.length-ut.length);break}}let I;return L==="px"&&e.defaultUnit!=="px"?I=b.in[e.defaultUnit]/e.defaultDPI:(I=b[L][e.defaultUnit],I<0&&(I=b[L].in*e.defaultDPI)),I*parseFloat(N)}function T(N){if(!(N.hasAttribute("transform")||N.nodeName==="use"&&(N.hasAttribute("x")||N.hasAttribute("y"))))return null;let L=w(N);return J.length>0&&L.premultiply(J[J.length-1]),U.copy(L),J.push(L),L}function w(N){let L=new Xt;if(N.nodeName==="use"&&(N.hasAttribute("x")||N.hasAttribute("y"))){let I=_(N.getAttribute("x")||0),G=_(N.getAttribute("y")||0);L.makeTranslation(I,G)}return N.hasAttribute("transform")&&x(N.getAttribute("transform"),L),L}function x(N,L){let I=at,G=N.split(")");for(let ht=G.length-1;ht>=0;ht--){let ut=G[ht].trim();if(ut==="")continue;let E=ut.indexOf("("),$=ut.length;if(E>0&&E<$){let B=ut.slice(0,E),nt=p(ut.slice(E+1));switch(I.identity(),B){case"translate":if(nt.length>=1){let it=nt[0],pt=0;nt.length>=2&&(pt=nt[1]),I.makeTranslation(it,pt)}break;case"rotate":if(nt.length>=1){let it=0,pt=0,mt=0;it=nt[0]*Math.PI/180,nt.length>=3&&(pt=nt[1],mt=nt[2]),W.makeTranslation(-pt,-mt),tt.makeRotation(it),vt.multiplyMatrices(tt,W),W.makeTranslation(pt,mt),I.multiplyMatrices(W,vt)}break;case"scale":if(nt.length>=1){let it=nt[0],pt=it;nt.length>=2&&(pt=nt[1]),I.makeScale(it,pt)}break;case"skewX":nt.length===1&&I.set(1,Math.tan(nt[0]*Math.PI/180),0,0,1,0,0,0,1);break;case"skewY":nt.length===1&&I.set(1,0,0,Math.tan(nt[0]*Math.PI/180),1,0,0,0,1);break;case"matrix":nt.length===6&&I.set(nt[0],nt[2],nt[4],nt[1],nt[3],nt[5],0,0,1);break}L.premultiply(I)}}return L}function M(N,L){function I(E){st.set(E.x,E.y,1).applyMatrix3(L),E.set(st.x,st.y)}function G(E){let $=E.xRadius,B=E.yRadius,nt=Math.cos(E.aRotation),it=Math.sin(E.aRotation),pt=new F($*nt,$*it,0),mt=new F(-B*it,B*nt,0),X=pt.applyMatrix3(L),Z=mt.applyMatrix3(L),bt=at.set(X.x,Z.x,0,X.y,Z.y,0,0,0,1),H=W.copy(bt).invert(),se=tt.copy(H).transpose().multiply(H).elements,Mt=D(se[0],se[1],se[4]),fe=Math.sqrt(Mt.rt1),Gt=Math.sqrt(Mt.rt2);if(E.xRadius=1/fe,E.yRadius=1/Gt,E.aRotation=Math.atan2(Mt.sn,Mt.cs),!((E.aEndAngle-E.aStartAngle)%(2*Math.PI)<Number.EPSILON)){let R=W.set(fe,0,0,0,Gt,0,0,0,1),S=tt.set(Mt.cs,Mt.sn,0,-Mt.sn,Mt.cs,0,0,0,1),rt=R.multiply(S).multiply(bt),St=yt=>{let{x:Et,y:Rt}=new F(Math.cos(yt),Math.sin(yt),0).applyMatrix3(rt);return Math.atan2(Rt,Et)};E.aStartAngle=St(E.aStartAngle),E.aEndAngle=St(E.aEndAngle),P(L)&&(E.aClockwise=!E.aClockwise)}}function ht(E){let $=k(L),B=O(L);E.xRadius*=$,E.yRadius*=B;let nt=$>Number.EPSILON?Math.atan2(L.elements[1],L.elements[0]):Math.atan2(-L.elements[3],L.elements[4]);E.aRotation+=nt,P(L)&&(E.aStartAngle*=-1,E.aEndAngle*=-1,E.aClockwise=!E.aClockwise)}let ut=N.subPaths;for(let E=0,$=ut.length;E<$;E++){let nt=ut[E].curves;for(let it=0;it<nt.length;it++){let pt=nt[it];pt.isLineCurve?(I(pt.v1),I(pt.v2)):pt.isCubicBezierCurve?(I(pt.v0),I(pt.v1),I(pt.v2),I(pt.v3)):pt.isQuadraticBezierCurve?(I(pt.v0),I(pt.v1),I(pt.v2)):pt.isEllipseCurve&&(wt.set(pt.aX,pt.aY),I(wt),pt.aX=wt.x,pt.aY=wt.y,C(L)?G(pt):ht(pt))}}}function P(N){let L=N.elements;return L[0]*L[4]-L[1]*L[3]<0}function C(N){let L=N.elements,I=L[0]*L[3]+L[1]*L[4];if(I===0)return!1;let G=k(N),ht=O(N);return Math.abs(I/(G*ht))>Number.EPSILON}function k(N){let L=N.elements;return Math.sqrt(L[0]*L[0]+L[1]*L[1])}function O(N){let L=N.elements;return Math.sqrt(L[3]*L[3]+L[4]*L[4])}function j(N){let L=N.elements,I=L[0]*L[4]-L[1]*L[3];return Math.sqrt(Math.abs(I))}function D(N,L,I){let G,ht,ut,E,$,B=N+I,nt=N-I,it=Math.sqrt(nt*nt+4*L*L);return B>0?(G=.5*(B+it),$=1/G,ht=N*$*I-L*$*L):B<0?ht=.5*(B-it):(G=.5*it,ht=-.5*it),nt>0?ut=nt+it:ut=nt-it,Math.abs(ut)>2*Math.abs(L)?($=-2*L/ut,E=1/Math.sqrt(1+$*$),ut=$*E):Math.abs(L)===0?(ut=1,E=0):($=-.5*ut/L,ut=1/Math.sqrt(1+$*$),E=$*ut),nt>0&&($=ut,ut=-E,E=$),{rt1:G,rt2:ht,cs:ut,sn:E}}let K=[],Q={},Y={},J=[],at=new Xt,W=new Xt,tt=new Xt,vt=new Xt,wt=new ft,st=new F,U=new Xt,et=new DOMParser().parseFromString(t,"image/svg+xml");return m(et),n(et.documentElement,{fill:"#000",fillOpacity:1,strokeOpacity:1,strokeWidth:1,strokeLineJoin:"miter",strokeLineCap:"butt",strokeMiterLimit:4}),{paths:K,gradients:Y,xml:et.documentElement}}static createFillMaterial(t){let e=t.userData.style;if(e.fill===void 0||e.fill==="none")return null;let n=t.color,s=null,r=yd.exec(e.fill);if(r){let o=t.userData.gradients&&t.userData.gradients[r[1]];s=p_(o,t)}let a=new Ae({opacity:e.fillOpacity*(e.opacity||1),transparent:!0,side:Ee,depthWrite:!1});return s!==null?a.map=s:a.color=n,a}static createStrokeMaterial(t){let e=t.userData.style;return e.stroke===void 0||e.stroke==="none"?null:(yd.test(e.stroke)&&console.warn("THREE.SVGLoader: Gradient strokes are not supported."),new Ae({color:new Vt().setStyle(e.stroke,Xr),opacity:e.strokeOpacity*(e.opacity||1),transparent:!0,side:Ee,depthWrite:!1}))}static createShapes(t){return console.warn("SVGLoader: createShapes() is deprecated. Use shapePath.toShapes() instead."),t.toShapes()}static getStrokeStyle(t,e,n,s,r){return t=t!==void 0?t:1,e=e!==void 0?e:"#000",n=n!==void 0?n:"miter",s=s!==void 0?s:"butt",r=r!==void 0?r:4,{strokeColor:e,strokeWidth:t,strokeLineJoin:n,strokeLineCap:s,strokeMiterLimit:r}}static pointsToStroke(t,e,n,s){let r=[],a=[],o=[];if(i.pointsToStrokeWithBuffers(t,e,n,s,r,a,o)===0)return null;let c=new de;return c.setAttribute("position",new Jt(r,3)),c.setAttribute("normal",new Jt(a,3)),c.setAttribute("uv",new Jt(o,2)),c}static pointsToStrokeWithBuffers(t,e,n,s,r,a,o,c){let l=new ft,u=new ft,h=new ft,d=new ft,f=new ft,m=new ft,y=new ft,g=new ft,p=new ft,v=new ft,b=new ft,_=new ft,T=new ft,w=new ft,x=new ft,M=new ft,P=new ft;n=n!==void 0?n:12,s=s!==void 0?s:.001,c=c!==void 0?c:0,t=ut(t);let C=t.length;if(C<2)return 0;let k=t[0].equals(t[C-1]),O,j=t[0],D,K=e.strokeWidth/2,Q=1/(C-1),Y=0,J,at,W,tt,vt=!1,wt=0,st=c*3,U=c*2;et(t[0],t[1],l).multiplyScalar(K),g.copy(t[0]).sub(l),p.copy(t[0]).add(l),v.copy(g),b.copy(p);for(let E=1;E<C;E++){O=t[E],E===C-1?k?D=t[1]:D=void 0:D=t[E+1];let $=l;if(et(j,O,$),h.copy($).multiplyScalar(K),_.copy(O).sub(h),T.copy(O).add(h),J=Y+Q,at=!1,D!==void 0){et(O,D,u),h.copy(u).multiplyScalar(K),w.copy(O).sub(h),x.copy(O).add(h),W=!0,h.subVectors(D,j),$.dot(h)<0&&(W=!1),E===1&&(vt=W),h.subVectors(D,O),h.normalize();let B=Math.abs($.dot(h));if(B>Number.EPSILON){let nt=K/B;h.multiplyScalar(-nt),d.subVectors(O,j),f.copy(d).setLength(nt).add(h),M.copy(f).negate();let it=f.length(),pt=d.length();d.divideScalar(pt),m.subVectors(D,O);let mt=m.length();if(m.divideScalar(mt),d.dot(M)<pt&&m.dot(M)<mt&&(at=!0),P.copy(f).add(O),M.add(O),at){let X=W?p:g,Z=(P.x-X.x)*(M.y-X.y)-(P.y-X.y)*(M.x-X.x);(W&&Z<0||!W&&Z>0)&&M.copy(X)}switch(tt=!1,at?W?(x.copy(M),T.copy(M)):(w.copy(M),_.copy(M)):L(),e.strokeLineJoin){case"bevel":I(W,at,J);break;case"round":G(W,at),W?N(O,_,w,J,0):N(O,x,T,J,1);break;default:let X=K*e.strokeMiterLimit/it;if(X<1)if(e.strokeLineJoin!=="miter-clip"){I(W,at,J);break}else G(W,at),W?(m.subVectors(P,_).multiplyScalar(X).add(_),y.subVectors(P,w).multiplyScalar(X).add(w),V(_,J,0),V(m,J,0),V(O,J,.5),V(O,J,.5),V(m,J,0),V(y,J,0),V(O,J,.5),V(y,J,0),V(w,J,0)):(m.subVectors(P,T).multiplyScalar(X).add(T),y.subVectors(P,x).multiplyScalar(X).add(x),V(T,J,1),V(m,J,1),V(O,J,.5),V(O,J,.5),V(m,J,1),V(y,J,1),V(O,J,.5),V(y,J,1),V(x,J,1));else at?(W?(V(p,Y,1),V(g,Y,0),V(P,J,0),V(p,Y,1),V(P,J,0),V(M,J,1)):(V(p,Y,1),V(g,Y,0),V(P,J,1),V(g,Y,0),V(M,J,0),V(P,J,1)),W?w.copy(P):x.copy(P)):W?(V(_,J,0),V(P,J,0),V(O,J,.5),V(O,J,.5),V(P,J,0),V(w,J,0)):(V(T,J,1),V(P,J,1),V(O,J,.5),V(O,J,.5),V(P,J,1),V(x,J,1)),tt=!0;break}}else L()}else L();!k&&E===C-1&&ht(t[0],v,b,W,!0,Y),Y=J,j=O,g.copy(w),p.copy(x)}if(!k)ht(O,_,T,W,!1,J);else if(at&&r){let E=P,$=M;vt!==W&&(E=M,$=P),W?(tt||vt)&&($.toArray(r,0),$.toArray(r,9),tt&&E.toArray(r,3)):(tt||!vt)&&($.toArray(r,3),$.toArray(r,9),tt&&E.toArray(r,0))}if(r){let E=[new ft,new ft,new ft],$=c*3;for(let B=$;B<st;B+=9)E[0].set(r[B],r[B+1]),E[1].set(r[B+3],r[B+4]),E[2].set(r[B+6],r[B+7]),Un.area(E)<0&&(r[B+3]=E[0].x,r[B+4]=E[0].y)}return wt;function et(E,$,B){return B.subVectors($,E),B.set(-B.y,B.x).normalize()}function V(E,$,B){r&&(r[st]=E.x,r[st+1]=E.y,r[st+2]=0,a&&(a[st]=0,a[st+1]=0,a[st+2]=1),st+=3,o&&(o[U]=$,o[U+1]=B,U+=2)),wt+=3}function N(E,$,B,nt,it){l.copy($).sub(E).normalize(),u.copy(B).sub(E).normalize();let pt=Math.PI,mt=l.dot(u);Math.abs(mt)<1&&(pt=Math.abs(Math.acos(mt))),pt/=n,h.copy($);for(let X=0,Z=n-1;X<Z;X++)d.copy(h).rotateAround(E,pt),V(h,nt,it),V(d,nt,it),V(E,nt,.5),h.copy(d);V(h,nt,it),V(B,nt,it),V(E,nt,.5)}function L(){V(p,Y,1),V(g,Y,0),V(_,J,0),V(p,Y,1),V(_,J,0),V(T,J,1)}function I(E,$,B){$?E?(V(p,Y,1),V(g,Y,0),V(_,J,0),V(p,Y,1),V(_,J,0),V(M,J,1),V(_,B,0),V(w,B,0),V(M,B,.5)):(V(p,Y,1),V(g,Y,0),V(T,J,1),V(g,Y,0),V(M,J,0),V(T,J,1),V(T,B,1),V(M,B,0),V(x,B,1)):E?(V(_,B,0),V(w,B,0),V(O,B,.5)):(V(T,B,1),V(x,B,0),V(O,B,.5))}function G(E,$){$&&(E?(V(p,Y,1),V(g,Y,0),V(_,J,0),V(p,Y,1),V(_,J,0),V(M,J,1),V(_,Y,0),V(O,J,.5),V(M,J,1),V(O,J,.5),V(w,Y,0),V(M,J,1)):(V(p,Y,1),V(g,Y,0),V(T,J,1),V(g,Y,0),V(M,J,0),V(T,J,1),V(T,Y,1),V(M,J,0),V(O,J,.5),V(O,J,.5),V(M,J,0),V(x,Y,1)))}function ht(E,$,B,nt,it,pt){switch(e.strokeLineCap){case"round":it?N(E,B,$,pt,.5):N(E,$,B,pt,.5);break;case"square":if(it)l.subVectors($,E),u.set(l.y,-l.x),h.addVectors(l,u).add(E),d.subVectors(u,l).add(E),nt?(h.toArray(r,3),d.toArray(r,0),d.toArray(r,9)):(h.toArray(r,3),o[7]===1?d.toArray(r,9):h.toArray(r,9),d.toArray(r,0));else{l.subVectors(B,E),u.set(l.y,-l.x),h.addVectors(l,u).add(E),d.subVectors(u,l).add(E);let mt=r.length;nt?(h.toArray(r,mt-3),d.toArray(r,mt-6),d.toArray(r,mt-12)):(d.toArray(r,mt-6),h.toArray(r,mt-3),d.toArray(r,mt-12))}break;default:break}}function ut(E){let $=!1;for(let nt=1,it=E.length-1;nt<it;nt++)if(E[nt].distanceTo(E[nt+1])<s){$=!0;break}if(!$)return E;let B=[];B.push(E[0]);for(let nt=1,it=E.length-1;nt<it;nt++)E[nt].distanceTo(E[nt+1])>=s&&B.push(E[nt]);return B.push(E[E.length-1]),B}}},yd=/^\s*url\(\s*(?:["']\s*)?#([^)'"\s]+)(?:\s*["'])?\s*\)\s*$/;function p_(i,t,e=256){if(!i||!Array.isArray(i.stops)||i.stops.length===0)return null;let n=t.userData.transform,s=i.gradientUnits==="objectBoundingBox",r=null;if(s&&(r=m_(t,n),r===null))return null;function a(h,d,f){f.set(h,d,1),i.gradientTransform&&f.applyMatrix3(i.gradientTransform),s&&f.set(r.minX+f.x*r.width,r.minY+f.y*r.height,1),n&&f.applyMatrix3(n)}let o=document.createElement("canvas"),c;if(i.type==="linearGradient"){o.width=e,o.height=1;let h=o.getContext("2d"),d=h.createLinearGradient(0,0,e,0);vd(d,i.stops),h.fillStyle=d,h.fillRect(0,0,e,1);let f=new F,m=new F;a(i.x1,i.y1,f),a(i.x2,i.y2,m);let y=m.x-f.x,g=m.y-f.y,p=y*y+g*g||1e-20,v=y/p,b=g/p,_=-(v*f.x+b*f.y);c=new Xt().set(v,b,_,0,0,.5,0,0,1)}else{let h=i.cx,d=i.cy,f=i.fx,m=i.fy,y=i.r;if(i.gradientTransform){let M=new F;M.set(h,d,1).applyMatrix3(i.gradientTransform),h=M.x,d=M.y,M.set(f,m,1).applyMatrix3(i.gradientTransform),f=M.x,m=M.y}if(s&&(h=r.minX+h*r.width,d=r.minY+d*r.height,f=r.minX+f*r.width,m=r.minY+m*r.height,y=y*Math.sqrt((r.width*r.width+r.height*r.height)/2)),y<=0)return null;o.width=e,o.height=e;let g=o.getContext("2d"),p=h-y,v=d-y,b=2*y,_=e/b;g.setTransform(_,0,0,_,-p*_,-v*_);let T=g.createRadialGradient(f,m,0,h,d,y);vd(T,i.stops),g.fillStyle=T,g.fillRect(p,v,b,b);let w=n?n.clone().invert():new Xt;c=new Xt().set(1/b,0,-p/b,0,1/b,-v/b,0,0,1).multiply(w)}let l=new Oi(o);l.colorSpace=Xr,l.flipY=!1,l.matrixAutoUpdate=!1,l.matrix=c;let u=i.spreadMethod==="reflect"?gs:i.spreadMethod==="repeat"?ms:mn;return l.wrapS=u,l.wrapT=u,l}function m_(i,t){let e=t?t.clone().invert():null,n=new ft,s=new Is;for(let r of i.subPaths)for(let a of r.getPoints())n.copy(a),e&&n.applyMatrix3(e),s.expandByPoint(n);return s.isEmpty()?null:{minX:s.min.x,minY:s.min.y,width:s.max.x-s.min.x,height:s.max.y-s.min.y}}function vd(i,t){let e=new Vt;for(let n of t){let s=n.color;if(n.opacity<1){e.setStyle(n.color,Xr);let r=/rgb\(([^)]+)\)/.exec(e.getStyle(Xr));r&&(s=`rgba(${r[1]},${n.opacity})`)}i.addColorStop(Math.max(0,Math.min(1,n.offset)),s)}}var bd=`<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" aria-hidden="true"><path fill="#3776ab" d="M12 1.5c-4.4 0-4.2 1.9-4.2 1.9v2h4.3v.6H6.2S3 5.7 3 10.1s2.8 4.2 2.8 4.2h1.7v-2.4s-.1-2.8 2.8-2.8h4.8s2.7 0 2.7-2.7V4.1S18.2 1.5 12 1.5zM9.7 3.3a.8.8 0 1 1 0 1.6.8.8 0 0 1 0-1.6z"/><path fill="#ffd343" d="M12 22.5c4.4 0 4.2-1.9 4.2-1.9v-2h-4.3V18h5.9s3.2.3 3.2-4.1-2.8-4.2-2.8-4.2h-1.7v2.4s.1 2.8-2.8 2.8H8.9s-2.7 0-2.7 2.7v2.3s-.4 2.6 5.8 2.6zm2.3-1.8a.8.8 0 1 1 0-1.6.8.8 0 0 1 0 1.6z"/></svg>
`;var Md={type:"change"},kc={type:"start"},wd={type:"end"},Qo=new xi,Sd=new on,x_=Math.cos(70*oi.DEG2RAD),Oe=new F,sn=2*Math.PI,me={NONE:-1,ROTATE:0,DOLLY:1,PAN:2,TOUCH_ROTATE:3,TOUCH_PAN:4,TOUCH_DOLLY_PAN:5,TOUCH_DOLLY_ROTATE:6},Bc=1e-6,Bs=class extends Lr{constructor(t,e=null){super(t,e),this.state=me.NONE,this.target=new F,this.cursor=new F,this.minDistance=0,this.maxDistance=1/0,this.minZoom=0,this.maxZoom=1/0,this.minTargetRadius=0,this.maxTargetRadius=1/0,this.minPolarAngle=0,this.maxPolarAngle=Math.PI,this.minAzimuthAngle=-1/0,this.maxAzimuthAngle=1/0,this.enableDamping=!1,this.dampingFactor=.05,this.enableZoom=!0,this.zoomSpeed=1,this.enableRotate=!0,this.rotateSpeed=1,this.keyRotateSpeed=1,this.enablePan=!0,this.panSpeed=1,this.screenSpacePanning=!0,this.keyPanSpeed=7,this.zoomToCursor=!1,this.autoRotate=!1,this.autoRotateSpeed=2,this.keys={LEFT:"ArrowLeft",UP:"ArrowUp",RIGHT:"ArrowRight",BOTTOM:"ArrowDown"},this.mouseButtons={LEFT:Mi.ROTATE,MIDDLE:Mi.DOLLY,RIGHT:Mi.PAN},this.touches={ONE:Si.ROTATE,TWO:Si.DOLLY_PAN},this.target0=this.target.clone(),this.position0=this.object.position.clone(),this.zoom0=this.object.zoom,this._cursorStyle="auto",this._domElementKeyEvents=null,this._lastPosition=new F,this._lastQuaternion=new ln,this._lastTargetPosition=new F,this._quat=new ln().setFromUnitVectors(t.up,new F(0,1,0)),this._quatInverse=this._quat.clone().invert(),this._spherical=new Ps,this._sphericalDelta=new Ps,this._scale=1,this._panOffset=new F,this._rotateStart=new ft,this._rotateEnd=new ft,this._rotateDelta=new ft,this._panStart=new ft,this._panEnd=new ft,this._panDelta=new ft,this._dollyStart=new ft,this._dollyEnd=new ft,this._dollyDelta=new ft,this._dollyDirection=new F,this._mouse=new ft,this._performCursorZoom=!1,this._pointers=[],this._pointerPositions={},this._controlActive=!1,this._onPointerMove=y_.bind(this),this._onPointerDown=__.bind(this),this._onPointerUp=v_.bind(this),this._onContextMenu=E_.bind(this),this._onMouseWheel=S_.bind(this),this._onKeyDown=w_.bind(this),this._onTouchStart=T_.bind(this),this._onTouchMove=A_.bind(this),this._onMouseDown=b_.bind(this),this._onMouseMove=M_.bind(this),this._interceptControlDown=C_.bind(this),this._interceptControlUp=R_.bind(this),this.domElement!==null&&this.connect(this.domElement),this.update()}set cursorStyle(t){this._cursorStyle=t,t==="grab"?this.domElement.style.cursor="grab":this.domElement.style.cursor="auto"}get cursorStyle(){return this._cursorStyle}connect(t){super.connect(t),this.domElement.addEventListener("pointerdown",this._onPointerDown),this.domElement.addEventListener("pointercancel",this._onPointerUp),this.domElement.addEventListener("contextmenu",this._onContextMenu),this.domElement.addEventListener("wheel",this._onMouseWheel,{passive:!1}),this.domElement.ownerDocument.addEventListener("keydown",this._interceptControlDown,{passive:!0,capture:!0}),this.domElement.style.touchAction="none"}disconnect(){this.state=me.NONE,this.domElement.removeEventListener("pointerdown",this._onPointerDown),this.domElement.ownerDocument.removeEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.removeEventListener("pointerup",this._onPointerUp),this.domElement.removeEventListener("pointercancel",this._onPointerUp),this.domElement.removeEventListener("wheel",this._onMouseWheel),this.domElement.removeEventListener("contextmenu",this._onContextMenu),this.stopListenToKeyEvents(),this.domElement.ownerDocument.removeEventListener("keydown",this._interceptControlDown,{capture:!0}),this.domElement.ownerDocument.removeEventListener("keyup",this._interceptControlUp,{capture:!0}),this._controlActive=!1,this._pointers.length=0,this._pointerPositions={},this.domElement.style.touchAction="",this.domElement.style.cursor="auto"}dispose(){this.disconnect()}getPolarAngle(){return this._spherical.phi}getAzimuthalAngle(){return this._spherical.theta}getDistance(){return this.object.position.distanceTo(this.target)}listenToKeyEvents(t){t.addEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=t}stopListenToKeyEvents(){this._domElementKeyEvents!==null&&(this._domElementKeyEvents.removeEventListener("keydown",this._onKeyDown),this._domElementKeyEvents=null)}saveState(){this.target0.copy(this.target),this.position0.copy(this.object.position),this.zoom0=this.object.zoom}reset(){this.target.copy(this.target0),this.object.position.copy(this.position0),this.object.zoom=this.zoom0,this.object.updateProjectionMatrix(),this.dispatchEvent(Md),this.update(),this.state=me.NONE}pan(t,e){this._pan(t,e),this.update()}dollyIn(t){this._dollyIn(t),this.update()}dollyOut(t){this._dollyOut(t),this.update()}rotateLeft(t){this._rotateLeft(t),this.update()}rotateUp(t){this._rotateUp(t),this.update()}update(t=null){let e=this.object.position;Oe.copy(e).sub(this.target),Oe.applyQuaternion(this._quat),this._spherical.setFromVector3(Oe),this.autoRotate&&this.state===me.NONE&&this._rotateLeft(this._getAutoRotationAngle(t)),this.enableDamping?(this._spherical.theta+=this._sphericalDelta.theta*this.dampingFactor,this._spherical.phi+=this._sphericalDelta.phi*this.dampingFactor):(this._spherical.theta+=this._sphericalDelta.theta,this._spherical.phi+=this._sphericalDelta.phi);let n=this.minAzimuthAngle,s=this.maxAzimuthAngle;isFinite(n)&&isFinite(s)&&(n<-Math.PI?n+=sn:n>Math.PI&&(n-=sn),s<-Math.PI?s+=sn:s>Math.PI&&(s-=sn),n<=s?this._spherical.theta=Math.max(n,Math.min(s,this._spherical.theta)):this._spherical.theta=this._spherical.theta>(n+s)/2?Math.max(n,this._spherical.theta):Math.min(s,this._spherical.theta)),this._spherical.phi=Math.max(this.minPolarAngle,Math.min(this.maxPolarAngle,this._spherical.phi)),this._spherical.makeSafe(),this.enableDamping===!0?this.target.addScaledVector(this._panOffset,this.dampingFactor):this.target.add(this._panOffset),this.target.sub(this.cursor),this.target.clampLength(this.minTargetRadius,this.maxTargetRadius),this.target.add(this.cursor);let r=!1;if(this.zoomToCursor&&this._performCursorZoom||this.object.isOrthographicCamera)this._spherical.radius=this._clampDistance(this._spherical.radius);else{let a=this._spherical.radius;this._spherical.radius=this._clampDistance(this._spherical.radius*this._scale),r=a!=this._spherical.radius}if(Oe.setFromSpherical(this._spherical),Oe.applyQuaternion(this._quatInverse),e.copy(this.target).add(Oe),this.object.lookAt(this.target),this.enableDamping===!0?(this._sphericalDelta.theta*=1-this.dampingFactor,this._sphericalDelta.phi*=1-this.dampingFactor,this._panOffset.multiplyScalar(1-this.dampingFactor)):(this._sphericalDelta.set(0,0,0),this._panOffset.set(0,0,0)),this.zoomToCursor&&this._performCursorZoom){let a=null;if(this.object.isPerspectiveCamera){let o=Oe.length();a=this._clampDistance(o*this._scale);let c=o-a;this.object.position.addScaledVector(this._dollyDirection,c),this.object.updateMatrixWorld(),r=!!c}else if(this.object.isOrthographicCamera){let o=new F(this._mouse.x,this._mouse.y,0);o.unproject(this.object);let c=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),this.object.updateProjectionMatrix(),r=c!==this.object.zoom;let l=new F(this._mouse.x,this._mouse.y,0);l.unproject(this.object),this.object.position.sub(l).add(o),this.object.updateMatrixWorld(),a=Oe.length()}else console.warn("WARNING: OrbitControls.js encountered an unknown camera type - zoom to cursor disabled."),this.zoomToCursor=!1;a!==null&&(this.screenSpacePanning?this.target.set(0,0,-1).transformDirection(this.object.matrix).multiplyScalar(a).add(this.object.position):(Qo.origin.copy(this.object.position),Qo.direction.set(0,0,-1).transformDirection(this.object.matrix),Math.abs(this.object.up.dot(Qo.direction))<x_?this.object.lookAt(this.target):(Sd.setFromNormalAndCoplanarPoint(this.object.up,this.target),Qo.intersectPlane(Sd,this.target))))}else if(this.object.isOrthographicCamera){let a=this.object.zoom;this.object.zoom=Math.max(this.minZoom,Math.min(this.maxZoom,this.object.zoom/this._scale)),a!==this.object.zoom&&(this.object.updateProjectionMatrix(),r=!0)}return this._scale=1,this._performCursorZoom=!1,r||this._lastPosition.distanceToSquared(this.object.position)>Bc||8*(1-this._lastQuaternion.dot(this.object.quaternion))>Bc||this._lastTargetPosition.distanceToSquared(this.target)>Bc?(this.dispatchEvent(Md),this._lastPosition.copy(this.object.position),this._lastQuaternion.copy(this.object.quaternion),this._lastTargetPosition.copy(this.target),!0):!1}_getAutoRotationAngle(t){return t!==null?sn/60*this.autoRotateSpeed*t:sn/60/60*this.autoRotateSpeed}_getZoomScale(t){let e=Math.abs(t*.01);return Math.pow(.95,this.zoomSpeed*e)}_rotateLeft(t){this._sphericalDelta.theta-=t}_rotateUp(t){this._sphericalDelta.phi-=t}_panLeft(t,e){Oe.setFromMatrixColumn(e,0),Oe.multiplyScalar(-t),this._panOffset.add(Oe)}_panUp(t,e){this.screenSpacePanning===!0?Oe.setFromMatrixColumn(e,1):(Oe.setFromMatrixColumn(e,0),Oe.crossVectors(this.object.up,Oe)),Oe.multiplyScalar(t),this._panOffset.add(Oe)}_pan(t,e){let n=this.domElement;if(this.object.isPerspectiveCamera){let s=this.object.position;Oe.copy(s).sub(this.target);let r=Oe.length();r*=Math.tan(this.object.fov/2*Math.PI/180),this._panLeft(2*t*r/n.clientHeight,this.object.matrix),this._panUp(2*e*r/n.clientHeight,this.object.matrix)}else this.object.isOrthographicCamera?(this._panLeft(t*(this.object.right-this.object.left)/this.object.zoom/n.clientWidth,this.object.matrix),this._panUp(e*(this.object.top-this.object.bottom)/this.object.zoom/n.clientHeight,this.object.matrix)):(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - pan disabled."),this.enablePan=!1)}_dollyOut(t){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale/=t:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_dollyIn(t){this.object.isPerspectiveCamera||this.object.isOrthographicCamera?this._scale*=t:(console.warn("WARNING: OrbitControls.js encountered an unknown camera type - dolly/zoom disabled."),this.enableZoom=!1)}_updateZoomParameters(t,e){if(!this.zoomToCursor)return;this._performCursorZoom=!0;let n=this.domElement.getBoundingClientRect(),s=t-n.left,r=e-n.top,a=n.width,o=n.height;this._mouse.x=s/a*2-1,this._mouse.y=-(r/o)*2+1,this._dollyDirection.set(this._mouse.x,this._mouse.y,1).unproject(this.object).sub(this.object.position).normalize()}_clampDistance(t){return Math.max(this.minDistance,Math.min(this.maxDistance,t))}_handleMouseDownRotate(t){this._rotateStart.set(t.clientX,t.clientY)}_handleMouseDownDolly(t){this._updateZoomParameters(t.clientX,t.clientY),this._dollyStart.set(t.clientX,t.clientY)}_handleMouseDownPan(t){this._panStart.set(t.clientX,t.clientY)}_handleMouseMoveRotate(t){this._rotateEnd.set(t.clientX,t.clientY),this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);let e=this.domElement;this._rotateLeft(sn*this._rotateDelta.x/e.clientHeight),this._rotateUp(sn*this._rotateDelta.y/e.clientHeight),this._rotateStart.copy(this._rotateEnd),this.update()}_handleMouseMoveDolly(t){this._dollyEnd.set(t.clientX,t.clientY),this._dollyDelta.subVectors(this._dollyEnd,this._dollyStart),this._dollyDelta.y>0?this._dollyOut(this._getZoomScale(this._dollyDelta.y)):this._dollyDelta.y<0&&this._dollyIn(this._getZoomScale(this._dollyDelta.y)),this._dollyStart.copy(this._dollyEnd),this.zoomToCursor&&(this._performCursorZoom=!0,this._dollyDirection.set(this._mouse.x,this._mouse.y,1).unproject(this.object).sub(this.object.position).normalize()),this.update()}_handleMouseMovePan(t){this._panEnd.set(t.clientX,t.clientY),this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd),this.update()}_handleMouseWheel(t){this._updateZoomParameters(t.clientX,t.clientY),t.deltaY<0?this._dollyIn(this._getZoomScale(t.deltaY)):t.deltaY>0&&this._dollyOut(this._getZoomScale(t.deltaY)),this.update()}_handleKeyDown(t){let e=!1;switch(t.code){case this.keys.UP:t.ctrlKey||t.metaKey||t.shiftKey?this.enableRotate&&this._rotateUp(sn*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,this.keyPanSpeed),e=!0;break;case this.keys.BOTTOM:t.ctrlKey||t.metaKey||t.shiftKey?this.enableRotate&&this._rotateUp(-sn*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(0,-this.keyPanSpeed),e=!0;break;case this.keys.LEFT:t.ctrlKey||t.metaKey||t.shiftKey?this.enableRotate&&this._rotateLeft(sn*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(this.keyPanSpeed,0),e=!0;break;case this.keys.RIGHT:t.ctrlKey||t.metaKey||t.shiftKey?this.enableRotate&&this._rotateLeft(-sn*this.keyRotateSpeed/this.domElement.clientHeight):this.enablePan&&this._pan(-this.keyPanSpeed,0),e=!0;break}e&&(t.preventDefault(),this.update())}_handleTouchStartRotate(t){if(this._pointers.length===1)this._rotateStart.set(t.pageX,t.pageY);else{let e=this._getSecondPointerPosition(t),n=.5*(t.pageX+e.x),s=.5*(t.pageY+e.y);this._rotateStart.set(n,s)}}_handleTouchStartPan(t){if(this._pointers.length===1)this._panStart.set(t.pageX,t.pageY);else{let e=this._getSecondPointerPosition(t),n=.5*(t.pageX+e.x),s=.5*(t.pageY+e.y);this._panStart.set(n,s)}}_handleTouchStartDolly(t){let e=this._getSecondPointerPosition(t),n=t.pageX-e.x,s=t.pageY-e.y,r=Math.sqrt(n*n+s*s);this._dollyStart.set(0,r)}_handleTouchStartDollyPan(t){this.enableZoom&&this._handleTouchStartDolly(t),this.enablePan&&this._handleTouchStartPan(t)}_handleTouchStartDollyRotate(t){this.enableZoom&&this._handleTouchStartDolly(t),this.enableRotate&&this._handleTouchStartRotate(t)}_handleTouchMoveRotate(t){if(this._pointers.length==1)this._rotateEnd.set(t.pageX,t.pageY);else{let n=this._getSecondPointerPosition(t),s=.5*(t.pageX+n.x),r=.5*(t.pageY+n.y);this._rotateEnd.set(s,r)}this._rotateDelta.subVectors(this._rotateEnd,this._rotateStart).multiplyScalar(this.rotateSpeed);let e=this.domElement;this._rotateLeft(sn*this._rotateDelta.x/e.clientHeight),this._rotateUp(sn*this._rotateDelta.y/e.clientHeight),this._rotateStart.copy(this._rotateEnd)}_handleTouchMovePan(t){if(this._pointers.length===1)this._panEnd.set(t.pageX,t.pageY);else{let e=this._getSecondPointerPosition(t),n=.5*(t.pageX+e.x),s=.5*(t.pageY+e.y);this._panEnd.set(n,s)}this._panDelta.subVectors(this._panEnd,this._panStart).multiplyScalar(this.panSpeed),this._pan(this._panDelta.x,this._panDelta.y),this._panStart.copy(this._panEnd)}_handleTouchMoveDolly(t){let e=this._getSecondPointerPosition(t),n=t.pageX-e.x,s=t.pageY-e.y,r=Math.sqrt(n*n+s*s);this._dollyEnd.set(0,r),this._dollyDelta.set(0,Math.pow(this._dollyEnd.y/this._dollyStart.y,this.zoomSpeed)),this._dollyOut(this._dollyDelta.y),this._dollyStart.copy(this._dollyEnd);let a=(t.pageX+e.x)*.5,o=(t.pageY+e.y)*.5;this._updateZoomParameters(a,o)}_handleTouchMoveDollyPan(t){this.enableZoom&&this._handleTouchMoveDolly(t),this.enablePan&&this._handleTouchMovePan(t)}_handleTouchMoveDollyRotate(t){this.enableZoom&&this._handleTouchMoveDolly(t),this.enableRotate&&this._handleTouchMoveRotate(t)}_addPointer(t){this._pointers.push(t.pointerId)}_removePointer(t){delete this._pointerPositions[t.pointerId];for(let e=0;e<this._pointers.length;e++)if(this._pointers[e]==t.pointerId){this._pointers.splice(e,1);return}}_isTrackingPointer(t){for(let e=0;e<this._pointers.length;e++)if(this._pointers[e]==t.pointerId)return!0;return!1}_trackPointer(t){let e=this._pointerPositions[t.pointerId];e===void 0&&(e=new ft,this._pointerPositions[t.pointerId]=e),e.set(t.pageX,t.pageY)}_getSecondPointerPosition(t){let e=t.pointerId===this._pointers[0]?this._pointers[1]:this._pointers[0];return this._pointerPositions[e]}_customWheelEvent(t){let e=t.deltaMode,n={clientX:t.clientX,clientY:t.clientY,deltaY:t.deltaY};switch(e){case 1:n.deltaY*=16;break;case 2:n.deltaY*=100;break}return t.ctrlKey&&!this._controlActive&&(n.deltaY*=10),n}};function __(i){this.enabled!==!1&&(this._pointers.length===0&&(this.domElement.setPointerCapture(i.pointerId),this.domElement.ownerDocument.addEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.addEventListener("pointerup",this._onPointerUp)),!this._isTrackingPointer(i)&&(this._addPointer(i),i.pointerType==="touch"?this._onTouchStart(i):this._onMouseDown(i),this._cursorStyle==="grab"&&(this.domElement.style.cursor="grabbing")))}function y_(i){this.enabled!==!1&&(i.pointerType==="touch"?this._onTouchMove(i):this._onMouseMove(i))}function v_(i){switch(this._removePointer(i),this._pointers.length){case 0:this.domElement.releasePointerCapture(i.pointerId),this.domElement.ownerDocument.removeEventListener("pointermove",this._onPointerMove),this.domElement.ownerDocument.removeEventListener("pointerup",this._onPointerUp),this.dispatchEvent(wd),this.state=me.NONE,this._cursorStyle==="grab"&&(this.domElement.style.cursor="grab");break;case 1:let t=this._pointers[0],e=this._pointerPositions[t];this._onTouchStart({pointerId:t,pageX:e.x,pageY:e.y});break}}function b_(i){let t;switch(i.button){case 0:t=this.mouseButtons.LEFT;break;case 1:t=this.mouseButtons.MIDDLE;break;case 2:t=this.mouseButtons.RIGHT;break;default:t=-1}switch(t){case Mi.DOLLY:if(this.enableZoom===!1)return;this._handleMouseDownDolly(i),this.state=me.DOLLY;break;case Mi.ROTATE:if(i.ctrlKey||i.metaKey||i.shiftKey){if(this.enablePan===!1)return;this._handleMouseDownPan(i),this.state=me.PAN}else{if(this.enableRotate===!1)return;this._handleMouseDownRotate(i),this.state=me.ROTATE}break;case Mi.PAN:if(i.ctrlKey||i.metaKey||i.shiftKey){if(this.enableRotate===!1)return;this._handleMouseDownRotate(i),this.state=me.ROTATE}else{if(this.enablePan===!1)return;this._handleMouseDownPan(i),this.state=me.PAN}break;default:this.state=me.NONE}this.state!==me.NONE&&this.dispatchEvent(kc)}function M_(i){switch(this.state){case me.ROTATE:if(this.enableRotate===!1)return;this._handleMouseMoveRotate(i);break;case me.DOLLY:if(this.enableZoom===!1)return;this._handleMouseMoveDolly(i);break;case me.PAN:if(this.enablePan===!1)return;this._handleMouseMovePan(i);break}}function S_(i){this.enabled===!1||this.enableZoom===!1||this.state!==me.NONE||(i.preventDefault(),this.dispatchEvent(kc),this._handleMouseWheel(this._customWheelEvent(i)),this.dispatchEvent(wd))}function w_(i){this.enabled!==!1&&this._handleKeyDown(i)}function T_(i){switch(this._trackPointer(i),this._pointers.length){case 1:switch(this.touches.ONE){case Si.ROTATE:if(this.enableRotate===!1){this.state=me.NONE;break}this._handleTouchStartRotate(i),this.state=me.TOUCH_ROTATE;break;case Si.PAN:if(this.enablePan===!1){this.state=me.NONE;break}this._handleTouchStartPan(i),this.state=me.TOUCH_PAN;break;default:this.state=me.NONE}break;case 2:switch(this.touches.TWO){case Si.DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchStartDollyPan(i),this.state=me.TOUCH_DOLLY_PAN;break;case Si.DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchStartDollyRotate(i),this.state=me.TOUCH_DOLLY_ROTATE;break;default:this.state=me.NONE}break;default:this.state=me.NONE}this.state!==me.NONE&&this.dispatchEvent(kc)}function A_(i){switch(this._trackPointer(i),this.state){case me.TOUCH_ROTATE:if(this.enableRotate===!1)return;this._handleTouchMoveRotate(i),this.update();break;case me.TOUCH_PAN:if(this.enablePan===!1)return;this._handleTouchMovePan(i),this.update();break;case me.TOUCH_DOLLY_PAN:if(this.enableZoom===!1&&this.enablePan===!1)return;this._handleTouchMoveDollyPan(i),this.update();break;case me.TOUCH_DOLLY_ROTATE:if(this.enableZoom===!1&&this.enableRotate===!1)return;this._handleTouchMoveDollyRotate(i),this.update();break;default:this.state=me.NONE}}function E_(i){this.enabled!==!1&&i.preventDefault()}function C_(i){i.key==="Control"&&(this._controlActive=!0,this.domElement.ownerDocument.addEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}function R_(i){i.key==="Control"&&(this._controlActive=!1,this.domElement.ownerDocument.removeEventListener("keyup",this._interceptControlUp,{passive:!0,capture:!0}))}function Td(i){return{surfaceY:1.85+(i.inputs.grossHeadM-5)/245*4.45,turbineFlow:i.turbineFlowM3s,bypassFlow:i.spillFlowM3s,load:i.plantMW/i.inputs.ratingMW,energized:i.plantMW>0}}var P_='<path d="M7 4.5v15l12-7.5z"/>',I_='<rect x="6" y="6" width="12" height="12" rx="1.2"/>';function ks(i,t,e="animation"){let n=`${t?"Stop":"Play"} ${e}`;i.title!==n&&(i.innerHTML=`<svg viewBox="0 0 24 24" width="19" height="19" fill="currentColor" aria-hidden="true" focusable="false">${t?I_:P_}</svg>`,i.title=n,i.setAttribute("aria-label",n)),i.style.width="40px",i.style.height="40px",i.style.padding="0",i.style.lineHeight="1",i.firstElementChild.style.verticalAlign="middle"}var tl=class{constructor(t,e=!0){this.scene=t,this.animate=e,this.parts=[],this.streams=[],this.electrical=[],this.concrete=this.mat("#c9cbc8",.88),this.cut=this.mat("#e4e3dc",.94),this.steel=this.mat("#667886",.32,.65),this.dark=this.mat("#263c49",.48,.5),this.copper=this.mat("#b97140",.35,.65),this.teal=this.mat("#267b80",.3,.45),this.water=this.mat("#168fb2",.22,.18)}mat(t,e=.6,n=0){return new Re({color:t,roughness:e,metalness:n,side:Ee})}mesh(t,e,n,s,r=this.scene){let a=new he(t,e);return a.position.set(...n),a.castShadow=!0,a.receiveShadow=!0,r.add(a),s&&(a.userData.label=s,this.parts.push(a)),a}box(t,e,n,s,r){return this.mesh(new Ce(...t),n,e,s,r)}cyl(t,e,n,s,r,a){return this.mesh(new Le(t,t,e,48),s,n,r,a)}bar(t,e,n,s,r){let a=new F(...t),o=new F(...e),c=o.clone().sub(a),l=this.mesh(new Le(n,n,c.length(),12),s,a.add(o).multiplyScalar(.5).toArray(),r);return l.quaternion.setFromUnitVectors(new F(0,1,0),c.normalize()),l}section(t,e,n,s,r){let a=new zi;return t.forEach(([o,c],l)=>l?a.lineTo(o,c):a.moveTo(o,c)),a.closePath(),this.mesh(new Vi(a,{depth:e,bevelEnabled:!1}),s,[0,0,n],r)}pipe(t,e,n,s="turbine"){let r=new tn(t.map(l=>new F(...l)),!1,"centripetal"),a=this.mat("#74969f",.32,.45);a.transparent=!0,a.opacity=.2,a.depthWrite=!1,this.mesh(new un(r,80,e,24,!1),a,[0,0,0],n),this.mesh(new un(r,80,e*.52,16,!1),this.water,[0,0,0],n);let o=[],c=this.mat("#c9faff",.15);for(let l=0;l<22;l++){let u=this.mesh(new De(e*.24,10,8),c,[0,0,0]);u.castShadow=!1,o.push(u)}this.streams.push({curve:r,particles:o,type:s,phase:0})}build(){this.landscape(),this.box([19,.55,7.4],[0,-.95,-.5],this.mat("#9caaa8"),"Foundation / bedrock"),this.reservoir=this.box([5.3,1,5.7],[-6.7,2,-1],this.water,"Reservoir \u2014 controlled water elevation");let t=[[-4,-.6],[.8,-.6],[-2.8,6.5],[-4,6.5]];this.section(t,4.7,-3.8,this.concrete,"Gravity dam"),this.section(t,.12,.9,this.cut,"Dam section"),this.box([1.35,.18,4.95],[-3.4,6.55,-1.35],this.cut,"Dam crest");for(let s of[-3.65,.8]){this.bar([-3.9,7,s],[-2.85,7,s],.025,this.steel,"Crest railing");for(let r of[-3.9,-3.35,-2.85])this.bar([r,6.65,s],[r,7,s],.022,this.steel)}this.pipe([[-4.45,1.2,1.5],[-3.6,1.2,1.5],[-2.5,1.1,1.5],[-1.1,.9,1.5],[.1,.8,1.5],[2.3,.8,1.5]],.4,"Submerged intake and pressurized penstock");for(let s=0;s<7;s++)this.bar([-4.45,.7,1+s*.16],[-4.45,1.7,1+s*.16],.03,this.steel,"Submerged intake trash rack");this.box([.12,.85,.9],[-4.05,1.2,1.5],this.steel,"Intake gate \u2014 open around exposed sectional water path"),this.bar([-3.75,1.7,1.5],[-3.75,6.8,1.5],.06,this.steel,"Intake control gate stem"),this.box([.4,.2,.5],[-3.75,6.8,1.5],this.dark,"Gate actuator"),this.box([5.2,.35,3.5],[3.5,-.3,-.1],this.concrete,"Powerhouse foundation"),this.box([5.2,3.8,.24],[3.5,1.8,-1.75],this.concrete,"Powerhouse"),this.box([.24,3.8,3.5],[6,1.8,-.1],this.concrete,"Powerhouse"),this.box([5.4,.22,1.4],[3.5,3.8,-1.15],this.cut,"Powerhouse roof cutaway"),this.box([1.2,.22,3.3],[5.4,2.1,-.05],this.cut,"Generator floor");for(let s=0;s<4;s++)this.box([.8,.65,.035],[1.6+s*1.05,2.8,-1.6],this.mat("#55828d",.3),"Powerhouse windows");this.cyl(1.05,.25,[2.5,.7,1.5],this.teal,"Turbine casing"),this.runner=new ce,this.runner.position.set(2.5,.95,1.5),this.scene.add(this.runner),this.cyl(.23,.32,[0,0,0],this.steel,"Turbine hub",this.runner);for(let s=0;s<11;s++){let r=s*Math.PI*2/11;this.box([.68,.12,.23],[Math.cos(r)*.55,0,Math.sin(r)*.55],s===0?this.copper:this.steel,"Turbine runner",this.runner).rotation.set(.25,-r+.45,.12)}this.cyl(.105,1.9,[2.5,1.9,1.5],this.steel,"Common turbine\u2013generator shaft"),this.mesh(new Le(.9,.9,.65,64,1,!0,Math.PI*.5,Math.PI),this.copper,[2.5,2.8,1.5],"Generator stator \u2014 cutaway");for(let s=0;s<18;s++){let r=Math.PI*.5+s*Math.PI/17;this.cyl(.045,.6,[2.5+.93*Math.sin(r),2.8,1.5+.93*Math.cos(r)],this.copper,"Stator windings")}this.rotor=new ce,this.rotor.position.set(2.5,2.65,1.5),this.scene.add(this.rotor),this.cyl(.64,.22,[0,0,0],this.dark,"Generator rotor",this.rotor);for(let s=0;s<8;s++){let r=s*Math.PI/4,a=this.box([.22,.28,.12],[Math.sin(r)*.53,.16,Math.cos(r)*.53],this.copper,"Generator rotor poles",this.rotor);a.rotation.y=r}this.pipe([[2.5,.7,1.5],[2.5,-.15,1.5],[3,-.45,1.5],[4.2,-.45,1.5],[6.6,-.45,1.5]],.43,"Draft tube \u2014 discharge"),this.box([3.6,.24,6],[7.6,-.6,-.6],this.water,"Tailrace / river"),this.section([[-3.4,6.35],[-2.7,6.35],[1.7,.1],[6.4,-.2],[6.4,-.4],[1.4,-.2]],1.05,-3.5,this.cut,"Spillway chute"),this.pipe([[-4.3,1.2,-3],[-2.5,.9,-3],[.2,.2,-3],[1.8,-.15,-3],[6.6,-.3,-3]],.18,"Gated low-level bypass \u2014 excess usable flow","spill"),this.box([1,.9,1],[4.6,4.3,-1.1],this.steel,"Step-up transformer");for(let s=0;s<8;s++)this.box([1.12,.055,1.08],[4.6,3.95+s*.1,-1.1],this.steel,"Transformer cooling fins");for(let s of[4.3,4.9])this.cyl(.065,.45,[s,5,-1.1],this.dark,"Transformer insulator");let e=7,n=-3.2;for(let s of[-1,1])this.bar([e+s*.65,0,n],[e+s*.2,5.7,n],.045,this.steel,"Transmission tower");for(let s=0;s<5;s+=.8)this.bar([e-.6+s*.075,s,n],[e+.6-(s+.8)*.075,s+.8,n],.026,this.steel,"Transmission tower"),this.bar([e+.6-s*.075,s,n],[e-.6+(s+.8)*.075,s+.8,n],.026,this.steel);this.bar([e-1,5.3,n],[e+1,5.3,n],.05,this.steel,"Transmission crossarm"),this.energyPath([[2.5,2.85,1.5],[3.6,2.85,1.5],[4.1,3.2,.7],[4.3,4.8,-1.1]],"Generator output bus \u2192 transformer");for(let s of[-.7,.7])this.energyPath([[4.6+s*.4,5.1,-1.1],[5.6+s*.6,4.85,-2.1],[e+s,5.3,n],[8.2+s,4.9,-4.2],[9+s,5.15,-5]],"Transformer \u2192 transmission grid");this.lamp=this.mat("#3fb6a0",.3),this.cyl(.12,.1,[4.6,4.82,-1.1],this.lamp,"Electrical output indicator")}landscape(){let t=this.mat("#708b75",.95),e=this.mat("#85918a",1),n=this.mat("#66584a",1),s=this.mat("#3e6654",1);this.box([21,.35,4],[0,-.8,-5.7],t,"River valley");let r=new ni(22,7,44,18);r.rotateX(-Math.PI/2);let a=(c,l)=>-.65+(5.8*Math.exp(-(((c+3)/3.3)**2))+3.9*Math.exp(-(((c-3)/2.8)**2))+3.4*Math.exp(-(((c+8)/2.8)**2)))*Math.max(0,1-(l/3.5)**2)*(1+.07*Math.sin(c*2+l)*Math.cos(l*3)),o=r.attributes.position;for(let c=0;c<o.count;c++){let l=o.getX(c),u=o.getZ(c);o.setY(c,a(l,u))}r.computeVertexNormals(),this.mesh(r,e,[0,0,-7],"Mountain catchment");for(let[c,l,u]of[[-9,-6.7,.9],[-7.8,-7,1.1],[-6.5,-6.4,1],[-5,-6.5,.9],[-3.6,-6,.9],[-2.1,-5.7,.8],[-.5,-5.3,.85],[1.2,-5.1,.8]]){let h=a(c,l+7);this.cyl(.07*u,.7*u,[c,h+.35*u,l],n,"Tree");for(let d=0;d<3;d++)this.mesh(new Tn((.48-d*.1)*u,.8*u,9),s,[c,h+(.85+d*.32)*u,l],"Tree")}}energyPath(t,e){let n=new tn(t.map(a=>new F(...a)),!1,"centripetal");this.mesh(new un(n,48,.035,8,!1),this.dark,[0,0,0],e);let s=new Re({color:"#ffc65a",emissive:"#ffad24",emissiveIntensity:1.8,roughness:.3}),r=[];for(let a=0;a<8;a++){let o=this.mesh(new De(.07,10,8),s,[0,0,0]);o.castShadow=!1,r.push(o)}this.electrical.push({curve:n,particles:r,material:s,phase:0})}update(t){this.result=t,this.state=Td(t);let e=this.state.surfaceY+.25;this.reservoir.scale.y=e,this.reservoir.position.y=e/2-.25,this.lamp.emissive.set("#29d6b0"),this.lamp.emissiveIntensity=this.state.load}streamMotion(t,e){let n=t.type==="spill"?this.result?.spillFlowM3s??0:e;return Math.min(.5,n*.025)}advance(t){let e=this.result?.turbineFlowM3s??0,n=this.animate?t*Math.min(7,e*1.3):0;this.runner.rotation.y+=n,this.rotor.rotation.y+=n;for(let s of this.streams){let r=this.streamMotion(s,e);this.animate&&(s.phase=(s.phase+t*r)%1),s.particles.forEach((a,o)=>{a.visible=r>0,a.position.copy(s.curve.getPointAt((o/s.particles.length+s.phase)%1)),a.position.z+=s.offset??(s.type==="spill"?.14:.32)})}for(let s of this.electrical)this.animate&&this.state?.energized&&(s.phase=(s.phase+t*(.08+.32*this.state.load))%1),s.material.emissiveIntensity=.6+2*(this.state?.load??0),s.particles.forEach((r,a)=>{r.visible=!!this.state?.energized,r.position.copy(s.curve.getPointAt((a/8+s.phase)%1))})}};var N_=Object.freeze({efficiency:.85,densityKgM3:1e3,gravityMS2:9.81,durationHours:6}),Ei=Object.freeze({grossHeadM:193,usableFlowM3s:10,headLossM:2,ratingMW:2}),Yr=Object.freeze({grossHeadM:[5,250],usableFlowM3s:[0,30],headLossM:[0,50],ratingMW:[.1,25]});function el(i,t){let e=Number(i);return Number.isFinite(e)?e:t}function nl(i,t,e){return Math.min(e,Math.max(t,i))}function U_(i={}){let t=nl(el(i.grossHeadM,Ei.grossHeadM),...Yr.grossHeadM),e=nl(el(i.usableFlowM3s,Ei.usableFlowM3s),...Yr.usableFlowM3s),n=Math.min(Yr.headLossM[1],Math.max(0,t-.5)),s=nl(el(i.headLossM,Ei.headLossM),Yr.headLossM[0],n),r=nl(el(i.ratingMW,Ei.ratingMW),...Yr.ratingMW);return{grossHeadM:t,usableFlowM3s:e,headLossM:s,ratingMW:r}}function Ad(i=Ei){let t=U_(i),{efficiency:e,densityKgM3:n,gravityMS2:s,durationHours:r}=N_,a=Math.max(0,t.grossHeadM-t.headLossM),o=n*s*t.usableFlowM3s*a/1e6,c=e*o,l=Math.min(c,t.ratingMW),u=a>0?t.ratingMW*1e6/(e*n*s*a):0,h=Math.min(t.usableFlowM3s,u),d=Math.max(0,t.usableFlowM3s-h);return{inputs:t,netHeadM:a,hydraulicMW:o,uncappedMW:c,plantMW:l,turbineFlowM3s:h,spillFlowM3s:d,energy6hMWh:l*r,clipped:c>t.ratingMW+1e-9}}var Ki=Math.PI/180,$r=8760,F_=287.05,zc=16/27;function Ed(i,t){return i*1e3/(F_*(t+273.15))}function O_(i,t,e){let n=Math.PI*e**2,s=t*n*i,r=i**2/2;return{area:n,massFlow:s,specificEnergy:r,windMW:s*r/1e6}}function B_(i,t,e){let n=i*Math.min(t,zc),s=n*e;return{rotorMW:n,electricalMW:s,drivetrainLossMW:n-s,betzMW:i*zc}}function k_(i,{cutIn:t,ratedSpeed:e,cutOut:n,ratedMW:s}){return i<=t||i>=n?0:i>=e?s:s*(i**3-t**3)/(e**3-t**3)}function il(i){if(i.outputMode!=="aero")return n=>k_(n,i);let e=.5*Ed(i.pressure,i.temperature)*Math.PI*i.radius**2*Math.min(i.cp,zc)*i.efficiency/1e8;return n=>n<=i.cutIn||n>=i.cutOut?0:Math.min(i.ratedMW,e*n**3)}function sl(i){let t=i.hub/i.reference,e=Math.log((i.hub-i.displacement)/i.roughness)/Math.log((i.reference-i.displacement)/i.roughness),n=i.heightMode==="power"?t**i.alpha:i.heightMode==="log"?e:1,s=i.speed*n,r=Ed(i.pressure,i.temperature),{area:a,massFlow:o,windMW:c}=O_(s,r,i.radius),{rotorMW:l,electricalMW:u}=B_(c,i.cp,i.efficiency/100),h=il(i)(s),d=i.outputMode==="aero"&&u>0?Math.max(i.cutIn,s*Math.cbrt(i.ratedMW/u)):i.ratedSpeed,f=s<=i.cutIn?"Below or at cut-in":s>=i.cutOut?"Cut-out shutdown":h===i.ratedMW?"Rated output":h>0?"Generating":"Zero generation",m=h>0?i.tsr*Math.min(s,d)/i.radius:0;return{speed:s,density:r,area:a,massFlow:o,windMW:c,rotorMW:l,aeroMW:u,gridMW:h,region:f,omega:m,rpm:m*60/(2*Math.PI),tsr:s?m*i.radius/s:null,torque:m?h*1e6/(i.efficiency/100*m):null,frictionVelocity:i.speed*.4/Math.log((i.reference-i.displacement)/i.roughness)}}function Cd(i){let t=l=>Math.exp(-((l/i.scale)**i.shape)),e=il(i),n=Math.ceil((i.cutOut-i.cutIn)/.05),s=(i.cutOut-i.cutIn)/n,r=0;for(let l=0;l<n;l++){let u=i.cutIn+l*s,h=u+s;r+=e((u+h)/2)*(t(u)-t(h))}let a=$r*r,o=i.availability/100*(1-i.wake/100)*(1-i.loss/100),c=a*o;return{meanMW:r,gross:a,retained:o,net:c,capacityFactor:c/($r*i.ratedMW)}}function Rd(i,t){let e=(Math.sin(i*Ki)+Math.sin(t*Ki))/2,n=(Math.cos(i*Ki)+Math.cos(t*Ki))/2,s=Math.hypot(e,n);return{length:s,degrees:s<1e-10?null:(Math.atan2(e,n)/Ki+360)%360}}function Gc(i,t){let e=(W,tt=0)=>new Re({color:W,metalness:tt,roughness:.59}),n=e("#e3e9eb"),s=e("#788a93",.48),r=e("#334752",.4),a=e("#be7545",.45),o=e("#657e62"),c=[];function l(W,tt,vt,wt,st=i){let U=new he(W,tt);return U.position.set(...vt),U.castShadow=U.receiveShadow=!0,wt&&(U.userData.label=wt,c.push(U)),st.add(U),U}let u=(W,tt,vt,wt,st)=>l(new Ce(...W),tt,vt,wt,st),h=(W,tt,vt,wt,st,U,et)=>l(new Le(W,tt,vt,28),wt,st,U,et);function d(W,tt,vt,wt,st){let U=new F(...W),V=new F(...tt).clone().sub(U),N=l(new Le(vt,vt,V.length(),8),wt,U.addScaledVector(V,.5).toArray(),st);return N.quaternion.setFromUnitVectors(new F(0,1,0),V.normalize()),N}u([17,.55,9.6],e("#aeb8a1"),[0,-.28,0],"Terrain section"),u([16.95,.09,9.55],o,[0,-.015,0],"Site terrain"),u([9.3,.06,1.65],e("#abb2a8"),[1.4,.06,.25],"Service pad");let f=[e("#476a55"),e("#58765b"),e("#749268")];for(let[W,tt,vt]of[[-6.6,-2.8,1.2],[-5,-3.6,.85],[-3.7,-3.65,1.05],[1.2,-3.8,.9],[2.75,-3.8,1.16],[6.9,-3.6,.9]]){h(.07,.11,vt,e("#6c5644"),[W,vt/2,tt],"Tree");for(let wt=0;wt<3;wt++)l(new Tn((.48-wt*.08)*vt,.82*vt,10),f[wt],[W,(1.04+wt*.28)*vt,tt],"Tree")}let m=-2.8,y=h(.27,.6,7.5,n,[m,3.85,0],"Tapered turbine tower");h(.81,.94,.27,s,[m,.17,0],"Tower foundation");let g=new ce;i.add(g),u([1.13,.19,2.1],n,[0,-.31,-.38],"Nacelle frame",g),u([.19,.78,2.1],n,[-.51,.05,-.38],"Nacelle cutaway",g),u([1.13,.19,.4],n,[0,.42,-1.2],"Nacelle roof section",g);function p(W,tt,vt,wt,st,U=g){let et=h(W,W,tt,wt,[0,.06,vt],st,U);return et.rotation.x=Math.PI/2,et}p(.12,.72,.41,s,"Low-speed shaft"),p(.39,.43,.01,r,"Gearbox"),p(.34,.13,.23,s,"Gearbox inspection flange"),p(.105,.46,-.29,s,"High-speed shaft"),p(.39,.49,-.66,a,"Generator");for(let W of[-.91,-.78,-.64,-.5,-.41])p(.408,.036,W,s,"Generator cooling fins");let v=new ce;v.position.set(0,.06,.82),g.add(v),l(new De(.37,28,14),n,[0,0,.12],"Rotor hub",v);let b=[[.28,.26,.09,0],[.72,.43,.105,.06],[1.48,.34,.085,.09],[2.35,.21,.057,.14],[3.23,.068,.024,.2],[3.36,.012,.008,.21]],_=[],T=[],w=12;for(let[W,tt,vt,wt]of b){let st=.28-W*.055;for(let U=0;U<w;U++){let et=U*2*Math.PI/w,V=Math.cos(et)*tt/2,N=Math.sin(et)*vt;_.push(wt+V*Math.cos(st)-N*Math.sin(st),W,V*Math.sin(st)+N*Math.cos(st))}}for(let W=0;W<b.length-1;W++)for(let tt=0;tt<w;tt++){let vt=W*w+tt,wt=W*w+(tt+1)%w;T.push(vt,vt+w,wt,wt,vt+w,wt+w)}let x=new de;x.setAttribute("position",new Jt(_,3)),x.setIndex(T),x.computeVertexNormals();let M=new ce;if(v.add(M),t){let W=new Ae({color:"#328baa",opacity:.65,transparent:!0,side:Ee,depthWrite:!1}),tt=new Ae({color:"#66a8ba",opacity:.14,transparent:!0,side:Ee,depthWrite:!1});l(new ki(3.36,64),tt,[0,0,-.11],"Rotor swept area",M).castShadow=!1,l(new Er(3.33,3.37,64),W,[0,0,-.1],"Edge of swept area",M).castShadow=!1}for(let W=0;W<3;W++){let tt=l(x,n,[0,0,0],"Rotor blade",M);tt.rotation.z=W*2*Math.PI/3}u([1.48,1.05,.95],r,[1.3,.63,.6],"Step-up transformer");for(let W=0;W<8;W++)u([1.65,.052,1.1],s,[1.3,.33+W*.097,.6],"Transformer cooling fins");let P=5.15,C=-1.25;for(let W of[-1,1]){for(let tt of[-1,1])d([P+W*.62,0,C+tt*.55],[P+W*.15,5.6,C+tt*.12],.038,r,"Transmission pylon");for(let tt=0;tt<5;tt+=.83){let vt=.62-tt*.083;d([P-W*vt,tt,C+.36],[P+W*(vt-.06),tt+.83,C+.29],.026,s,"Pylon bracing")}}d([P-1.22,4.94,C],[P+1.22,4.94,C],.05,r,"Transmission crossarm");let k=[],O=new De(.088,10,8),j=new Ae({color:"#ffc554"});function D(W,tt,vt=6){let wt=new tn(W.map(U=>new F(...U)),!1,"centripetal");l(new un(wt,56,.033,8,!1),r,[0,0,0],tt);let st=Array.from({length:vt},()=>{let U=l(O,j,[0,0,0]);return U.castShadow=!1,U});k.push({curve:wt,pulses:st})}D([[m+.79,.19,.34],[m+1.12,.14,.49],[-.26,.14,.58],[.56,.25,.6]],"Generator feeder to step-up transformer",4);for(let W of[-.86,0,.86])h(.055,.055,.32,s,[1.3+W*.54,1.28,.6],"Transformer bushing"),h(.062,.062,.32,n,[P+W,4.79,C],"Line insulator"),D([[1.3+W*.54,1.44,.6],[3.25+W*.6,2.5,-.25],[P+W,4.63,C],[6.65+W*.52,4.12,-1.9],[8.2+W*.34,4.4,-2.6]],"Grid transmission conductor");let K=new ce;i.add(K);let Q=Array.from({length:10},(W,tt)=>{let vt=u([.043,.042,.55],new Ae({color:tt%2?"#328baa":"#66a8ba"}),[0,0,0],"",K);return vt.castShadow=!1,vt}),Y=0,J=0;function at(W,tt,vt){vt&&(v.rotation.z=(v.rotation.z-W*tt.omega)%(2*Math.PI),Y=(Y+W*Math.min(.9,tt.gridMW*.3))%1,J=(J+W*tt.speed/18)%1),k.forEach(({curve:wt,pulses:st})=>st.forEach((U,et)=>{U.visible=tt.gridMW>0,U.position.copy(wt.getPointAt((et/st.length+Y)%1))})),Q.forEach((wt,st)=>{wt.visible=tt.speed>0,wt.position.set((st%5-2)*.9,(st%3-1)*.8,4.5-8.7*((st/10+J)%1))})}return{advance:at,parts:c,column:y,head:g,blades:M,air:K,towerX:m,rotor:v,powerLines:k,airflow:Q}}function Pd(i,t,{showSweep:e=!1}={}){let n=new wn;n.background=new Vt("#e6eff3");let s=new Gn({antialias:!0});s.setPixelRatio(Math.min(devicePixelRatio,2)),s.shadowMap.enabled=!0,s.toneMapping=Pn,i.append(s.domElement);let r=new En(-10,10,8,-8,.1,100);r.position.set(12,11,23);let a=new Bs(r,s.domElement);a.target.set(0,5,0),Object.assign(a,{enablePan:!1,enableZoom:!1,minPolarAngle:.55,maxPolarAngle:1.47}),a.update(),n.add(new An(15136255,8357224,2.4));let o=new Cn(16772557,3);o.position.set(-7,16,9),o.castShadow=!0,o.shadow.mapSize.set(1024,1024),Object.assign(o.shadow.camera,{left:-12,right:12,top:12,bottom:-12}),o.shadow.normalBias=.035,n.add(o);let{advance:c,parts:l,column:u,head:h,blades:d,air:f,towerX:m,rotor:y,powerLines:g,airflow:p}=Gc(n,e),v,b,_=0,T=!0,w=!0,x;function M(D,K,Q){v=K,b=Q;let Y=[D.hub,D.radius,D.direction].join(":");if(Y!==x){let J=7.5+(D.hub-90)/44,at=e?D.radius/60:Math.min(1.24,Math.max(.64,D.radius/58)),W=D.direction*Ki;u.scale.y=J/7.5,u.position.y=J/2+.1,h.position.set(m,J+.35,0),h.rotation.y=W,d.scale.setScalar(at),f.position.copy(h.position),f.rotation.y=W,x=Y}i.dataset.windRegion=K.region,T=!0}let P=new ii;function C(D){let K=i.getBoundingClientRect();P.setFromCamera(new ft(2*(D.clientX-K.left)/K.width-1,1-2*(D.clientY-K.top)/K.height),r);let Q=P.intersectObjects(l,!1)[0];t.hidden=!Q,Q&&(t.textContent=Q.object.userData.label,t.style.left=Math.max(6,Math.min(K.width-180,D.clientX-K.left+9))+"px",t.style.top=Math.max(5,D.clientY-K.top-40)+"px")}let k=()=>{t.hidden=!0};i.addEventListener("pointermove",C),i.addEventListener("pointerdown",C),i.addEventListener("pointerleave",k),a.addEventListener("change",()=>{T=!0});let O=new ResizeObserver(()=>{let{clientWidth:D,clientHeight:K}=i;if(!D||!K)return;let Q=Math.max(7.8,10*K/D);Object.assign(r,{left:-Q*D/K,right:Q*D/K,top:Q,bottom:-Q}),r.updateProjectionMatrix(),s.setSize(D,K,!1),T=!0}),j=new IntersectionObserver(([D])=>{w=D.isIntersecting,T=!0});return O.observe(i),j.observe(i),s.setAnimationLoop(D=>{let K=Math.min(.05,(D-_)/1e3);_=D,!(!v||!w||!b&&!T)&&(c(K,v,b),i.dataset.rotorAngle=String(y.rotation.z),s.render(n,r),T=!1)}),{update:M,destroy(){s.setAnimationLoop(null),O.disconnect(),j.disconnect(),a.dispose(),i.removeEventListener("pointermove",C),i.removeEventListener("pointerdown",C),i.removeEventListener("pointerleave",k);let D=new Set,K=new Set;n.traverse(Q=>{Q.geometry&&D.add(Q.geometry),Q.material&&K.add(Q.material)}),D.forEach(Q=>Q.dispose()),K.forEach(Q=>Q.dispose()),s.dispose(),s.domElement.remove()}}}var rl=`:host {
  --ink: #153b36;
  --muted: #5f716e;
  --line: #cbdad5;
  --surface: #ffffff;
  --paper: #f4f8f6;
  --green: #0f826c;
  --blue: #237fb6;
  --cyan: #48b7d5;
  --amber: #d78b23;
  --danger: #ad4d35;
  color: var(--ink);
  display: block;
  width: 100%;
  max-width: 100%;
  min-width: 0;
  overflow: hidden;
  font: 15px/1.5 Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}
.engineering-card { margin: 0; position: relative; overflow: hidden; border: 1px solid #cbd4d9; border-radius: 14px; background: #edf1f4; }
#plant-scene { height: clamp(400px, 48vw, 680px); position: relative; }
#plant-scene canvas { width: 100%; height: 100%; display: block; touch-action: none; }
.engineering-card figcaption { padding: 12px 16px; font-size: 12px; color: #526571; background: white; border-top: 1px solid #d5dee2; }
.engineering-card .head-ruler { position: static; width: auto; display: flex; gap: 12px; border: 0; border-radius: 0; padding: 10px 16px; }
.engineering-card .head-track { display: none; }
.plant-tooltip { position: absolute; z-index: 5; max-width: 175px; background: #183440; color: white; border-radius: 6px; padding: 8px 10px; font-size: 12px; pointer-events: none; }
.motion-toggle { position: absolute; left: 12px; top: 12px; background: white; color: #183440; border: 1px solid #bdcbd3; border-radius: 6px; padding: 8px 12px; cursor: pointer; }
@media(max-width:560px) { #plant-scene { height: 360px; } .engineering-card .head-ruler { flex-wrap: wrap; } }
sub { font-size: .72em; line-height: 0; vertical-align: sub; }
* { box-sizing: border-box; }
button, input { font: inherit; }
.hydro-lab { width: 100%; max-width: 100%; min-width: 0; border: 1px solid var(--line); border-radius: 18px; background: var(--paper); overflow: hidden; box-shadow: 0 16px 42px rgb(27 64 55 / 10%); }
.lab-head { padding: 22px 24px 14px; background: linear-gradient(135deg, #e8f6f1, #edf7fb); border-bottom: 1px solid var(--line); }
.lab-kicker { color: var(--green); font-size: 12px; font-weight: 800; letter-spacing: .11em; text-transform: uppercase; }
h2 { margin: 4px 0 5px; font-family: Georgia, "Times New Roman", serif; font-size: clamp(24px, 4vw, 36px); font-weight: 500; line-height: 1.15; overflow-wrap: anywhere; }
.lab-head p { max-width: 850px; margin: 0; color: var(--muted); }
.controls { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 14px; padding: 18px 22px; background: var(--surface); border-bottom: 1px solid var(--line); }
.control { min-width: 0; }
.control-line { display: flex; align-items: baseline; justify-content: space-between; gap: 8px; margin-bottom: 6px; }
.control label { font-weight: 700; color: var(--ink); }
.control output { font: 700 13px/1.2 ui-monospace, SFMono-Regular, Consolas, monospace; white-space: nowrap; }
input[type="range"] { width: 100%; accent-color: var(--blue); cursor: pointer; }
input[type="range"]:focus-visible { outline: 3px solid #65aee0; outline-offset: 3px; border-radius: 8px; }
.stage-grid { display: grid; grid-template-columns: minmax(0, 1.7fr) minmax(310px, .7fr); gap: 16px; padding: 18px; }
.stage-grid > *, .charts > *, .lab-head, .controls, .lab-foot { min-width: 0; max-width: 100%; }
.scene-card, .metrics-card, .chart-card { border: 1px solid var(--line); border-radius: 14px; background: var(--surface); overflow: hidden; }
.scene-card { min-height: 590px; position: relative; background: linear-gradient(180deg, #e8f7fb 0 58%, #d9eddd 58% 100%); }
.scene { position: absolute; inset: 0; overflow: hidden; }
.scene canvas { width: 100%; height: 100%; display: block; touch-action: none; }
.scene-labels { position: absolute; inset: 0; pointer-events: none; overflow: hidden; }
.scene-label { position: absolute; left: 0; top: 0; padding: 4px 8px; border: 1px solid rgb(22 65 57 / 22%); border-radius: 6px; background: rgb(255 255 255 / 92%); color: var(--ink); box-shadow: 0 2px 8px rgb(20 50 45 / 14%); font-size: 10px; font-weight: 800; white-space: nowrap; }
.scene-label::after { content: ""; position: absolute; left: 50%; top: 100%; width: 1px; height: 10px; background: rgb(21 59 54 / 45%); }
.scene-label.flow-label { color: #075e83; border-color: rgb(35 127 182 / 35%); }
.scene-label.water-label { color: #08769d; background: rgb(235 251 255 / 94%); }
.fallback { width: 100%; height: 100%; object-fit: contain; }
.scene-help { position: absolute; right: 12px; bottom: 10px; z-index: 2; margin: 0; padding: 5px 9px; border-radius: 8px; background: rgb(255 255 255 / 82%); color: var(--muted); font-size: 11px; pointer-events: none; }
.head-ruler { position: absolute; left: 12px; bottom: 10px; z-index: 2; width: 132px; padding: 9px; border-radius: 10px; background: rgb(255 255 255 / 90%); border: 1px solid var(--line); font-size: 11px; }
.head-ruler strong { display: block; font-size: 12px; }
.head-track { height: 75px; width: 10px; margin: 7px 8px 6px 3px; float: left; border-radius: 8px; background: linear-gradient(to top, var(--amber) 0 var(--loss-pct), var(--blue) var(--loss-pct) 100%); }
.head-ruler span { display: block; color: var(--muted); }
.metrics-card { padding: 16px; }
.power-card { padding: 14px; border-radius: 12px; background: #eaf5fb; border: 1px solid #c5e1ef; }
.power-card .eyebrow { color: var(--muted); font-size: 12px; }
.power-card strong { display: block; color: var(--blue); font: 800 clamp(28px, 5vw, 42px)/1.1 ui-monospace, SFMono-Regular, Consolas, monospace; }
.status { display: inline-block; margin-top: 7px; padding: 3px 8px; border-radius: 999px; font-size: 12px; font-weight: 800; }
.status.clipped { color: #7b331f; background: #ffe3d8; }
.status.open { color: #14684f; background: #d9f3e8; }
.metric-list { display: grid; gap: 0; margin: 13px 0; }
.metric { display: flex; justify-content: space-between; gap: 12px; padding: 9px 2px; border-bottom: 1px solid #e2ebe7; }
.metric span { color: var(--muted); }
.metric strong { text-align: right; font-family: ui-monospace, SFMono-Regular, Consolas, monospace; }
.equations { margin-top: 15px; }
.equations h3 { margin: 0 0 8px; font-size: 15px; }
.equation { margin: 6px 0; padding: 8px 10px; border-left: 3px solid var(--green); background: #f5faf8; font: 12px/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
.assumptions { margin: 13px 0 0; color: var(--muted); font-size: 12px; }
.charts { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; padding: 0 18px 18px; }
.chart-card { padding: 13px; }
.chart-card h3 { margin: 0 0 2px; font-size: 15px; }
.chart-card p { margin: 0 0 4px; color: var(--muted); font-size: 12px; }
.chart-card svg { display: block; width: 100%; height: auto; min-height: 230px; overflow: visible; }
.grid { stroke: #e1e9e6; stroke-width: 1; }
.axis { stroke: #80938e; stroke-width: 1.3; }
.tick, .axis-label, .legend, .bar-label, .bar-value { fill: var(--muted); font-size: 12px; }
.axis-label, .legend, .bar-value { font-weight: 700; }
.curve { fill: none; stroke-width: 3; }
.curve.uncapped { stroke: var(--blue); }
.curve.plant { stroke: var(--green); }
.curve.cap { stroke: var(--danger); stroke-dasharray: 7 5; }
.live-point { fill: var(--amber); stroke: #fff; stroke-width: 3; }
.live-label { fill: var(--ink); font-size: 12px; font-weight: 800; }
.cap-text { fill: var(--danger); }
.bar-track { fill: #e5ece9; }
.bar.hydraulic { fill: var(--cyan); }
.bar.uncapped { fill: var(--green); }
.bar.plant { fill: var(--amber); }
.energy-band { fill: #e7f3ed; stroke: #c3ddd1; }
.energy-label { fill: var(--ink); font-size: 14px; font-weight: 800; }
.lab-foot { padding: 12px 20px 16px; color: var(--muted); font-size: 12px; text-align: center; }
.lab-foot a { color: var(--blue); font-weight: 700; }
@media (max-width: 900px) {
  .controls { grid-template-columns: 1fr 1fr; }
  .stage-grid { grid-template-columns: 1fr; }
  .scene-card { min-height: 560px; }
  .charts { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
  .lab-head { padding: 18px 16px 12px; }
  .controls { display: block; padding: 14px 16px; }
  .control + .control { margin-top: 16px; }
  .stage-grid, .charts { display: block; width: 100%; padding: 12px; }
  .scene-card, .metrics-card, .chart-card { width: 100%; max-width: 100%; min-width: 0; }
  .metrics-card, .chart-card { margin-top: 12px; }
  .charts { padding-top: 0; }
  .scene-card { min-height: 500px; }
  .scene-label { font-size: 9px; padding: 3px 6px; }
  .scene-label::after { height: 7px; }
  .head-ruler { transform: scale(.88); transform-origin: bottom left; }
  .scene-help { max-width: 150px; text-align: right; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { scroll-behavior: auto !important; animation: none !important; transition: none !important; }
}
@media print {
  .controls, .scene-help { display: none; }
  .hydro-lab { box-shadow: none; }
}

.cutaway-card {
  container-type: inline-size;
  margin: 0;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--surface);
  overflow: hidden;
}
.plant-cutaway { display: block; width: 100%; height: auto; }
.component-target { outline: none; cursor: help; }
.component-hit { fill: transparent; stroke: transparent; stroke-width: 2; pointer-events: all; }
.component-tooltip { opacity: 0; pointer-events: none; transition: opacity .12s; }
.component-tooltip rect { fill: #173745; stroke: #e1edf1; stroke-width: 1; }
.component-tooltip text { fill: white; font: 500 18px Inter, sans-serif; }
.component-target:hover .component-tooltip,
.component-target:focus-visible .component-tooltip,
.component-target[aria-pressed="true"] .component-tooltip { opacity: 1; }
.component-target:hover .component-hit,
.component-target:focus-visible .component-hit { fill: #bce6f4; fill-opacity: .12; stroke: #63a8bb; }
.cutaway-card figcaption { padding: 12px 16px 0; color: var(--muted); font-size: 13px; }
.cutaway-card .head-ruler {
  position: static;
  display: flex;
  flex-wrap: wrap;
  gap: 4px 15px;
  align-items: baseline;
  width: auto;
  margin: 10px 16px 14px;
  padding: 9px 11px;
  transform: none;
  background: #f2f8f7;
}
.cutaway-card .head-track { width: 25px; height: 5px; margin: 0; align-self: center; }
.cutaway-label { fill: #18343b; }
.cutaway-label path { fill: none; stroke: #38535a; stroke-width: 1.6; }
.cutaway-label text {
  paint-order: stroke;
  stroke: #f3f6ee;
  stroke-width: 4px;
  stroke-linejoin: round;
  font-size: 19px;
  font-weight: 750;
}
.cutaway-marker, .cutaway-key { display: none; }
.cutaway-marker circle { fill: #f7fbf9; stroke: #164655; stroke-width: 2.5; }
.cutaway-marker text {
  fill: #163640;
  font-size: 19px;
  font-weight: 800;
  text-anchor: middle;
  dominant-baseline: central;
}
.cutaway-key {
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px 10px;
  margin: 0;
  padding: 12px;
  list-style: none;
  border-bottom: 1px solid var(--line);
  font-size: 12px;
}
.cutaway-key li { display: flex; gap: 7px; align-items: baseline; }
.cutaway-key span { min-width: 19px; text-align: center; color: #154859; font-weight: 800; }
.grid-lines { color: var(--grid-active, #354e49); }
.water-motion {
  fill: none;
  stroke-linecap: round;
  stroke-linejoin: round;
  stroke-dasharray: 11 35;
  animation: water-travel var(--turbine-speed, 3s) linear infinite;
}
#cutaway-spill-motion { animation-duration: var(--spill-speed, 2s); }
@keyframes water-travel { to { stroke-dashoffset: -46; } }
.orbit-view { margin: 0 0 8px; }
.orbit-view summary { cursor: pointer; color: #20556c; font-size: 13px; font-weight: 700; padding: 6px 10px; }
.orbit-view summary:focus-visible { outline: 2px solid var(--blue); outline-offset: 2px; }
.orbit-view .show-diagram, .orbit-view[open] .show-3d, .orbit-view[open] + .cutaway-card { display: none; }
.orbit-view[open] .show-diagram { display: inline; }
.orbit-view .scene-card { margin-top: 10px; }
@container (max-width: 610px) {
  .cutaway-label { display: none; }
  .cutaway-marker { display: inline; }
  .cutaway-key { display: grid; }
}
`;var al=`.solar-control-panel, .solar-analysis { padding: 14px 18px; display: grid; gap: 10px; }
.solar-lab fieldset, .solar-lab details { min-width: 0; margin: 0; padding: 10px 13px; background: #fff; border: 1px solid var(--line); border-radius: 10px; }
.solar-lab legend, .solar-lab summary { font-weight: 750; }
.solar-lab summary { cursor: pointer; }
.solar-controls { display: grid; grid-template-columns: repeat(auto-fit, minmax(min(100%, 210px), 1fr)); gap: 10px 16px; }
.solar-lab details .solar-controls { margin-top: 12px; }
.solar-lab .control-line { flex-wrap: wrap; gap: 3px 8px; }
.solar-lab .control label { font-size: 13px; }
.solar-lab input[type="range"] { display: block; margin: 4px 0 2px; }
.solar-stage { display: grid; grid-template-columns: minmax(0, 1.55fr) minmax(270px, 1fr); align-items: start; gap: 12px; padding: 0 18px; }
#solar-scene { height: clamp(300px, 44vw, 495px); }
#solar-scene canvas { display: block; width: 100%; height: 100%; touch-action: pan-y; }
.solar-lab .chart-card svg { min-height: 0; max-height: 260px; }
.solar-lab svg text { font-size: 11px; fill: #405950; }
.solar-lab .equation { font: 13px/1.55 system-ui, sans-serif; }
@media (max-width: 760px) {
  .solar-stage { grid-template-columns: minmax(0, 1fr); }
}
@media (max-width: 420px) {
  .solar-control-panel, .solar-analysis, .solar-stage { padding-inline: 10px; }
}
`;var Id=`#wind-scene { position: relative; height: clamp(290px, 30cqw, 440px); }
#wind-scene canvas { display: block; width: 100%; height: 100%; }
.wind-lab { container-type: inline-size; }
@container (max-width: 740px) {
  .wind-lab .solar-stage { grid-template-columns: minmax(0, 1fr); }
  #wind-scene { height: clamp(285px, 77cqw, 450px); }
}
.wind-selects { display: grid; grid-template-columns: repeat(auto-fit,minmax(min(100%,260px),1fr)); gap: 10px; }
.wind-selects label, .wind-selects select { display: block; min-width: 0; font: inherit; }
.wind-selects label { font-weight: 700; }
.wind-selects select { width: 100%; margin-top: 4px; border: 1px solid var(--line); border-radius: 7px; padding: 7px; color: var(--ink); background: white; }

.wind-lab .solar-stage { grid-template-columns: minmax(0, 1.15fr) minmax(0, 1fr) minmax(0, 1fr); }
.wind-lab .solar-stage > * { min-width: 0; max-width: 100%; }
@container (max-width: 900px) {
  .wind-lab .solar-stage { grid-template-columns: minmax(0, 1fr); }
  #wind-scene { height: clamp(285px, 77cqw, 450px); }
}
`;var Zr=[["Wind and turbine",[["speed","Wind speed at input height",0,35,.1,8,"m/s"],["direction","Wind from (clockwise from north)",0,360,1,12,"\xB0"],["radius","Blade radius R",15,100,1,58,"m"],["hub","Hub height z",35,180,1,90,"m"]]],["Air density and height correction",[["pressure","Air pressure p",75,104,.1,101.3,"kPa"],["temperature","Air temperature T",-20,45,1,15,"\xB0C"],["reference","Measurement height z<sub>r</sub>",10,120,1,10,"m"],["alpha","Shear exponent \u03B1",0,.45,.01,.14,""],["roughness","Roughness length z<sub>0</sub>",.01,1,.01,.05,"m"],["displacement","Displacement height d",0,8,.1,0,"m"]]],["Rotor, drivetrain and power curve",[["cp","Power coefficient C<sub>p</sub>",.05,.59,.01,.42,""],["efficiency","Drivetrain efficiency \u03B7<sub>drive</sub>",60,100,1,95,"%"],["tsr","Target tip-speed ratio below rated",3,10,.1,7,""],["ratedMW","Rated electrical power P<sub>r</sub>",.1,12,.1,3,"MW"],["cutIn","Cut-in speed v<sub>in</sub>",1,6,.1,3,"m/s"],["ratedSpeed","Rated speed v<sub>r</sub> (course curve)",7,18,.1,12,"m/s"],["cutOut","Cut-out speed v<sub>out</sub>",19,35,.1,25,"m/s"],["hours","Time at current wind speed \u0394t",.25,24,.25,2,"h"]]],["Annual energy and losses",[["shape","Weibull shape k",1,4,.1,2,""],["scale","Weibull scale c at hub height",3,18,.1,8.5,"m/s"],["availability","Availability",50,100,1,97,"%"],["wake","Wake loss",0,35,1,5,"%"],["loss","Electrical and other site losses",0,35,1,3,"%"]]],["Wind measurement examples",[["sigma","Wind-speed standard deviation \u03C3<sub>v</sub>",0,10,.1,1.2,"m/s"],["bearingA","First direction observation",0,360,1,350,"\xB0"],["bearingB","Second direction observation",0,360,1,10,"\xB0"]]]],Dd=Zr.flatMap(([,i])=>i),Hc=Object.fromEntries(Dd.map(([i,,,,,t])=>[i,t])),ye=(i,t=2)=>i===null?"undefined":i.toFixed(t);function Vc(i){return i.map(([t,e,n,s,r,a])=>`<div class="control">
    <div class="control-line"><label for="${t}">${e}</label><output id="${t}-value" aria-hidden="true"></output></div>
    <input id="${t}" type="range" min="${n}" max="${s}" step="${r}" value="${a}">
  </div>`).join("")}function H_(i,t,e){let n=il(t),s=270,r=d=>53+460*d/s,a=d=>160-132*d/12,o=t.outputMode==="aero"?n(t.cutIn+1e-7):0,c=[[t.cutIn,o]];for(let d=1;d<=120;d++){let f=t.cutIn+(t.cutOut-t.cutIn)*d/120;c.push([f,n(Math.min(f,t.cutOut-1e-7))])}let l=r(t.cutOut),u=a(c.at(-1)[1]),h=[[t.cutIn,"Cut-in"],[t.cutOut,"Cut-out"]];t.outputMode==="curve"&&h.splice(1,0,[t.ratedSpeed,"Rated"]),i.innerHTML=`<path class="axis" fill="none" d="M53 16V160H513"/>
    <text x="4" y="32">12 MW</text>
    <text x="16" y="164">0 MW</text><text x="53" y="190">0</text><text x="513" y="190" text-anchor="end">270</text><text x="514" y="180" text-anchor="end">Hub-height wind (m/s)</text>
    ${h.map(([d,f])=>`<path stroke="#91a6a5" stroke-dasharray="4 4" d="M${r(d)} 16V160"/>
      <text x="${r(d)}" y="176" text-anchor="middle">${f}</text>`).join("")}
    <path stroke="#087660" stroke-width="3" d="M${r(0)} ${a(0)}H${r(t.cutIn)}"/>
    <polyline fill="none" stroke="#087660" stroke-width="3" points="${c.map(([d,f])=>`${r(d)},${a(f)}`).join(" ")}"/>
    ${o>0?`<circle cx="${r(t.cutIn)}" cy="${a(o)}" r="4" fill="#fff" stroke="#087660" stroke-width="1.5"/>
      <circle cx="${r(t.cutIn)}" cy="${a(0)}" r="4" fill="#087660"/>`:""}
    <path stroke="#087660" stroke-width="3" d="M${l} ${a(0)}H${r(s)}"/>
    <circle cx="${l}" cy="${u}" r="4" fill="#fff" stroke="#087660" stroke-width="1.5"/>
    <circle cx="${l}" cy="${a(0)}" r="4" fill="#087660"/>
    <circle fill="#c77616" stroke="#fff" stroke-width="2" cx="${r(e.speed)}" cy="${a(e.gridMW)}" r="6"/>`}function W_(i){let t=i.shadowRoot??i.attachShadow({mode:"open"}),e=document.createElement("style");e.textContent=rl+al+Id;let n=document.createElement("div");n.innerHTML=`<article class="hydro-lab solar-lab wind-lab" aria-labelledby="wind-title">
    <header class="lab-head"><span class="lab-kicker">Interactive engineering diagram</span>
      <h2 id="wind-title">From moving air to grid electricity</h2>
      <p>Change the wind and follow energy through the rotor, generator, transformer and transmission line.</p></header>
    <div class="solar-control-panel">
      <fieldset><legend>Wind and turbine</legend><div class="solar-controls">${Vc(Zr[0][1])}</div></fieldset>
      <div class="wind-selects"><label>Wind speed input
        <select id="heightMode"><option value="hub">At hub height</option><option value="power">At measurement height \xB7 power law</option>
        <option value="log">At measurement height \xB7 neutral log law</option></select></label>
        <label>Electrical output model
        <select id="outputMode"><option value="curve">Course power curve</option><option value="aero">Fixed C\u209A with equipment limits</option></select></label></div>
      ${Zr.slice(1,4).map(([h,d])=>`<details><summary>${h}</summary><div class="solar-controls">${Vc(d)}</div></details>`).join("")}
    </div>
    <div class="solar-stage">
      <figure class="engineering-card">
        <div id="wind-scene" aria-label="Isometric wind turbine cutaway, step-up transformer and transmission pylon">
          <div class="plant-tooltip" id="wind-tooltip" hidden></div>
          <button class="motion-toggle" id="wind-play" type="button"></button>
        </div>
        <figcaption>Drag to orbit. Hover or tap a part to identify it. Blue streaks show wind direction. Gold pulses trace the electrical route to the grid. Geometry and pulse speed are schematic.</figcaption>
      </figure>
      <section class="metrics-card" aria-label="Wind turbine output">
        <div class="power-card"><span class="eyebrow">Electrical power to grid</span><strong id="grid"></strong><span class="status" id="region"></span></div>
        <div class="metric-list" id="wind-metrics"></div>
      </section>
      <section class="chart-card"><h3>Electrical power curve</h3><p>The orange dot marks the current hub-height wind speed. At cut-out, the filled point marks zero output.</p>
        <svg id="wind-chart" viewBox="0 0 535 192" role="img" aria-label="Electrical power versus hub-height wind speed"></svg></section>
    </div>
    <div class="solar-analysis">
      <details open><summary>Live equations</summary><div id="wind-equations"></div></details>
      <details><summary>${Zr[4][0]}</summary><div class="solar-controls">${Vc(Zr[4][1])}</div><div id="wind-measurements"></div></details>
      <p class="assumptions">Air density, blade radius, and drivetrain efficiency change electrical output in the fixed-C\u209A model. The course curve uses the specified electrical rating and speed thresholds. Annual energy uses a separate hub-height Weibull distribution and applies availability and site losses once. The rotor stops at cut-out in this teaching model.</p>
    </div>
  </article>`,t.replaceChildren(e,n);let s=h=>t.getElementById(h),r={...Hc,heightMode:"hub",outputMode:"curve"},a,o=!matchMedia("(prefers-reduced-motion: reduce)").matches;try{a=Pd(s("wind-scene"),s("wind-tooltip"))}catch{s("wind-scene").textContent="The 3D view requires WebGL. The equations, controls and power curve are available below."}let c=h=>`<div class="equation">${h}</div>`;function l(){let h=sl(r),d=Cd(r),f=Rd(r.bearingA,r.bearingB);s("grid").textContent=ye(h.gridMW,3)+" MW";let m=s("region");m.textContent=h.region,m.className=`status ${h.gridMW?"open":"clipped"}`,s("wind-metrics").innerHTML=[["Hub-height wind",`${ye(h.speed,1)} m/s`],["Rotor speed",`${ye(h.rpm,1)} rpm`],[`Energy over ${r.hours} h`,`${ye(r.hours*h.gridMW,3)} MWh`],["Net annual energy (Weibull)",`${ye(d.net,0)} MWh`],["Net annual capacity factor",`${ye(d.capacityFactor*100,1)}%`]].map(([p,v])=>`<div class="metric"><span>${p}</span><strong>${v}</strong></div>`).join("");let y=r.heightMode==="power"?`v = v<sub>ref</sub>(z/z<sub>r</sub>)<sup>\u03B1</sup> = ${r.speed} \xD7 (${r.hub}/${r.reference})<sup>${r.alpha}</sup>`:r.heightMode==="log"?`u<sub>*</sub> = \u03BAv<sub>ref</sub>/ln[(z<sub>r</sub>\u2212d)/z<sub>0</sub>] = ${ye(h.frictionVelocity,3)} m/s \xB7 v = (u<sub>*</sub>/\u03BA) ln[(z\u2212d)/z<sub>0</sub>], with \u03BA = 0.4`:"v = entered hub-height wind speed",g=r.outputMode==="aero"?"P<sub>grid</sub> = min(P<sub>e</sub>, P<sub>r</sub>) during operation, zero at/below cut-in and at/above cut-out":"P<sub>grid</sub> = P<sub>r</sub>(v\xB3\u2212v<sub>in</sub>\xB3)/(v<sub>r</sub>\xB3\u2212v<sub>in</sub>\xB3) on the ramp, P<sub>r</sub> on the plateau, zero at/below cut-in and at/above cut-out";s("wind-equations").innerHTML=[`${y} \u2192 hub wind ${ye(h.speed,2)} m/s`,`\u03C1 = p/(R<sub>d</sub>T) = ${ye(r.pressure*1e3,0)}/[287.05 \xD7 ${ye(r.temperature+273.15)}] \u2248 ${ye(h.density,3)} kg/m\xB3`,`A = \u03C0R\xB2 = ${ye(h.area,1)} m\xB2 \xB7 \u1E41 = \u03C1Av = ${ye(h.massFlow,0)} kg/s \xB7 P<sub>wind</sub> = \xBD\u03C1Av\xB3 = ${ye(h.windMW,3)} MW`,`Fixed-coefficient estimate: P<sub>rotor</sub> = C<sub>p</sub>P<sub>wind</sub> = ${ye(h.rotorMW,3)} MW \xB7 C<sub>p,max</sub> = 16/27 \xB7 P<sub>e</sub> = \u03B7<sub>drive</sub>P<sub>rotor</sub> = ${ye(h.aeroMW,3)} MW before equipment limits.`,`${g}. Current electrical output: ${ye(h.gridMW,3)} MW.`,`\u03A9 = 2\u03C0n/60 = ${ye(h.omega,3)} rad/s \xB7 \u03BB = \u03A9R/v = ${ye(h.tsr)} \xB7 \u03C4 = P<sub>shaft</sub>/\u03A9 = ${h.torque===null?"undefined when stopped":ye(h.torque/1e3,1)+" kN\xB7m"} (shaft power estimated from grid output and drivetrain efficiency)`,`E = \u03A3 P<sub>i</sub>\u0394t<sub>i</sub> = ${ye(h.gridMW,3)} \xD7 ${r.hours} \u2248 ${ye(r.hours*h.gridMW,3)} MWh for the current interval`,`Weibull F(v) = 1 \u2212 exp[\u2212(v/c)<sup>k</sup>], f(v) = (k/c)(v/c)<sup>k\u22121</sup> exp[\u2212(v/c)<sup>k</sup>]. Bin probability = F(b) \u2212 F(a). Mean power \u2248 \u03A3P(v<sub>mid</sub>)p<sub>i</sub> = ${ye(d.meanMW,3)} MW.`,`AEP<sub>gross</sub> = ${$r} \xD7 mean power = ${ye(d.gross,0)} MWh \xB7 AEP<sub>net</sub> = AEP<sub>gross</sub> \xD7 availability \xD7 (1 \u2212 wake loss) \xD7 (1 \u2212 site loss) = ${ye(d.net,0)} MWh`,`CF<sub>net</sub> = AEP<sub>net</sub>/(P<sub>r</sub> \xD7 ${$r} h) = ${ye(d.capacityFactor*100,1)}%`].map(c).join(""),s("wind-measurements").innerHTML=[`TI = \u03C3<sub>v</sub>/v\u0304 = ${h.speed>0?ye(r.sigma/h.speed*100,1)+"%":"undefined at zero mean speed"} using hub-height input as the example mean.`,`Circular mean = atan2(mean sin \u03B8, mean cos \u03B8) = ${f.degrees===null?"undefined for opposing directions":ye(f.degrees,1)+"\xB0"} \xB7 resultant length = ${ye(f.length,3)}.`].map(c).join(""),H_(s("wind-chart"),r,h),a?.update(r,h,o)}for(let[h,,,,,,d]of Dd){let f=s(h),m=()=>{r[h]=Number(f.value);let y=`${f.value} ${d}`.trim();s(h+"-value").textContent=y,f.setAttribute("aria-valuetext",y)};m(),f.oninput=()=>{m(),l()}}for(let h of["heightMode","outputMode"])s(h).onchange=d=>{r[h]=d.target.value,l()};let u=s("wind-play");if(u){let h=()=>{ks(u,o)};h(),u.onclick=()=>{o=!o,h(),l()}}return l(),{destroy(){a?.destroy(),t.replaceChildren()}}}var Ld=document.getElementById("wind-lab-root");Ld&&W_(Ld);var en=Math.PI/180,ol=1e3;function X_(i,t){let e=i*en,n=t*en;return[Math.sin(e)*Math.sin(n),Math.cos(e),Math.sin(e)*Math.cos(n)]}function q_(i,t){let e=i.reduce((n,s,r)=>n+s*t[r],0);return Math.max(-1,Math.min(1,e))}function Nd(i,t){return t>1e-12?i*t:0}function Y_(i,t,e,n,s){let r=Nd(i,n[1])+t,a=q_(n,s),o=Nd(i,a),c=t*(1+s[1])/2,l=e*r*(1-s[1])/2;return{ghi:r,incidence:a,beam:o,sky:c,ground:l,poa:o+c+l}}function $_(i,t,e){return t+i/800*(e-20)}function Z_(i,t){return Math.max(0,1+t*(i-25))}function J_(...i){return i.reduce((t,e)=>t*(1-e),1)}function K_(i,t,e){let n=i*t,s=Math.min(n,e);return{potential:n,ac:s,clip:n-s,conversionLoss:i-n}}function ll(i,t,e){let n=t*en,s=e*en,r=(i-12)*15*en;return[-Math.cos(s)*Math.sin(r),Math.sin(n)*Math.sin(s)+Math.cos(n)*Math.cos(s)*Math.cos(r),Math.cos(n)*Math.sin(s)-Math.sin(n)*Math.cos(s)*Math.cos(r)]}function Jr(i,t=i.hour){let e=ll(t,i.latitude,i.declination),n=Math.asin(Math.max(-1,Math.min(1,e[1])))/en,s=X_(i.tilt,i.azimuth),r=Math.max(0,e[1]),a=i.dni*Math.min(1,4*r),o=i.dhi*r,{ghi:c,incidence:l,beam:u,sky:h,ground:d,poa:f}=Y_(a,o,i.albedo,e,s),m=i.area*ol*i.efficiency/100,y=$_(f,i.ambient,i.noct),g=Z_(y,i.coefficient/100),p=f*i.area*i.efficiency/100,v=J_(i.soiling/100,i.wiring/100),b=p*g,_=K_(b*v,i.inverterEfficiency/100,i.rating);return{sun:e,normal:s,elevation:n,incidence:l,dni:a,dhi:o,ghi:c,beam:u,sky:h,ground:d,poa:f,rated:m,cell:y,tempFactor:g,areaPower:p,retained:v,dc:b,..._}}function Wc(i){let t=.08333333333333333,e=Array.from({length:288},(c,l)=>Jr(i,(l+.5)*t)),n=c=>e.reduce((l,u)=>l+u[c],0)*t/1e3,s=n("ac"),r=n("poa"),a=s/(e[0].rated/1e3),o=r/(ol/1e3);return{samples:e,energy:s,irradiation:r,yieldFinal:a,yieldRef:o,pr:o?a/o:null,cf:s/(i.rating/1e3*24),clipping:n("clip")}}function Ud(i,{load:t,storage:e,batteryPower:n,batteryEfficiency:s}){let r=.08333333333333333,a=s/100,o=0,c=0,l=0,u=0;for(let{ac:h}of i){let d=Math.min(h,t),f=Math.min(h-d,n,Math.max(0,(e-o)/(a*r))),m=Math.min(t-d,n,Math.max(0,o*a/r));o+=r*(a*f-m/a),c+=r*(t-d-m),l+=r*(h-d-f),u+=r*(d+m)}return{stored:o/1e3,unmet:c/1e3,surplus:l/1e3,used:u/1e3}}function Fd({firstYear:i,degradation:t,years:e,capital:n,maintenance:s,discount:r}){let a=0,o=0,c=n,l;for(let u=1;u<=e;u++){let h=(1+r/100)**u;l=i*(1-t/100)**(u-1),a+=l,o+=l/h,c+=s/h}return{energy:a,lastYear:l,lcoe:c/o}}function Od(i,t,e){let n=new ce;i([3.8,.16,2.85],e,[0,0,0],n),i([3.66,.045,2.71],t("#142740",.28),[0,.083,0],n);let s=["#173963","#1a416e","#204a77"].map(r=>t(r,.27,.26));for(let r=0;r<6;r++)for(let a=0;a<8;a++){let o=(a-3.5)*.45,c=(r-2.5)*.442;i([.426,.019,.418],s[(r+a*2)%3],[o,.114,c],n)}return n}function Xc(i){let t=(b,_=.7,T=0)=>new Re({color:b,roughness:_,metalness:T});function e(b,_,T,w=i){let x=new he(b,_);return x.position.set(...T),x.castShadow=x.receiveShadow=!0,w.add(x),x}let n=(b,_,T,w)=>e(new Ce(...b),_,T,w),s=t("#8798a4",.35,.55);n([13,.48,9],t("#9aaa8e"),[0,-.26,0]),n([12.85,.08,8.85],t("#687e61"),[0,-.025,0]),n([4.4,.12,3.6],t("#b8b8a7"),[0,.07,0]),n([.3,2,.3],s,[0,1,0]),n([.8,.16,.8],s,[0,.19,0]);let r=new ce,a=Od(n,t,s);r.add(a),r.position.y=2.05,i.add(r);let o=n([.95,.72,.59],t("#dce2e3"),[3.13,.45,-.9]),c=new Ae({color:"#425a63"});n([.43,.14,.018],c,[3.13,.57,-1.2]);let l=n([.62,.88,.53],t("#627681"),[5.1,.45,-.9]),u=new Ae({color:"#56686d"});n([.22,.08,.02],u,[5.1,.69,-1.17]);let h=[[[0,1.9,0],[0,.2,0],[1.6,.14,-.45],[2.7,.16,-.9],o.position.toArray()],[o.position.toArray(),[3.7,.14,-.9],[4.7,.14,-.9],l.position.toArray()]].map(b=>{let _=new tn(b.map(w=>new F(...w)),!1,"centripetal");e(new un(_,36,.044,8,!1),t("#31414a"),[0,0,0]);let T=e(new De(.086,12,8),new Ae({color:"#ffc153"}),[0,0,0]);return T.castShadow=!1,{curve:_,pulse:T}}),d=e(new De(.42,32,16),new Ae({color:"#ffbd3e"}),[0,0,0]);d.castShadow=!1;let f=new ws(new de,new Hi({color:"#ba8e48",dashSize:.17,gapSize:.11})),m=new ws(new de,new Fi({color:"#edaa32",transparent:!0}));i.add(f,m);let y=new F(0,2.05,0),g=b=>new F(...b).multiplyScalar(6.1).add(y),p=0;function v(b,_,T){T&&(p=(p+b*.48)%1),h.forEach(({curve:w,pulse:x},M)=>{x.visible=(M?_.ac:_.dc)>0,x.position.copy(w.getPointAt((p+M*.5)%1))})}return{advance:v,rack:r,panel:a,path:f,rays:m,sunPoint:g,sun:d,led:c,output:u,wires:h}}function Bd(i){let t=new wn,e=new Vt("#e4eef2"),n=new Vt("#26394f");t.background=e.clone();let s=new Gn({antialias:!0});s.setPixelRatio(Math.min(devicePixelRatio,2)),s.shadowMap.enabled=!0,s.toneMapping=Pn,i.append(s.domElement);let r=new En(-8,8,6,-6,.1,100);r.position.set(13,10,-16);let a=new Bs(r,s.domElement);a.target.set(0,2.7,0),Object.assign(a,{enablePan:!1,maxPolarAngle:Math.PI/2.1,minPolarAngle:.42,minZoom:.8,maxZoom:1.85,enableZoom:!1}),a.update();let o=new An(15070207,6584663,2);t.add(o);let c=new Cn(16768926,3);c.castShadow=!0,c.shadow.mapSize.set(1024,1024),Object.assign(c.shadow.camera,{left:-8,right:8,top:8,bottom:-8}),c.shadow.normalBias=.035,t.add(c);let{advance:l,rack:u,panel:h,path:d,rays:f,sunPoint:m,sun:y,led:g,output:p,wires:v}=Xc(t),b,_=0,T,w=!1,x=!0,M=!0;function P(O,j,D){w=D,T=j,u.rotation.y=O.azimuth*en,h.rotation.x=O.tilt*en;let K=`${O.latitude},${O.declination}`;if(K!==b){let Y=[];for(let J=0;J<120;J++){let at=ll(J/5,O.latitude,O.declination),W=ll((J+1)/5,O.latitude,O.declination);at[1]>0&&W[1]>0&&Y.push(m(at),m(W))}d.geometry.dispose(),d.geometry=new de().setFromPoints(Y),d.computeLineDistances(),b=K}y.position.copy(m(j.sun)),c.position.copy(y.position),c.intensity=j.dni/310,o.intensity=j.elevation>0?1.4+j.dhi/240:.75,t.background.copy(e).lerp(n,j.elevation>0?0:.86),y.visible=j.elevation>0,f.visible=j.beam>0,f.material.opacity=Math.min(.9,.2+j.beam/1100);let Q=[];h.updateWorldMatrix(!0,!1);for(let Y of[-1.15,0,1.15])for(let J of[-.7,.7]){let at=h.localToWorld(new F(Y,.15,J));Q.push(at.clone().addScaledVector(new F(...j.sun),3.2),at)}f.geometry.dispose(),f.geometry=new de().setFromPoints(Q),g.color.set(j.dc>0?"#67db9f":"#425a63"),p.color.set(j.ac>0?"#ffc153":"#56686d"),M=!0}let C=new ResizeObserver(()=>{let{clientWidth:O,clientHeight:j}=i;if(!O||!j)return;let D=Math.max(5.9,7.8*j/O);r.left=-D*O/j,r.right=-r.left,r.top=D,r.bottom=-D,r.updateProjectionMatrix(),s.setSize(O,j,!1),M=!0}),k=new IntersectionObserver(([O])=>{x=O.isIntersecting,M=!0});return a.addEventListener("change",()=>{M=!0}),C.observe(i),k.observe(i),s.setAnimationLoop(O=>{let j=Math.min(.05,(O-_)/1e3);_=O,!(!x||!T||!M&&!w)&&(l(j,T,w),s.render(t,r),M=!1)}),{update:P,destroy(){s.setAnimationLoop(null),C.disconnect(),k.disconnect(),a.dispose();let O=new Set;t.traverse(j=>{j.geometry?.dispose(),j.material&&O.add(j.material)}),O.forEach(j=>j.dispose()),s.dispose(),s.domElement.remove()}}}var Kr=[["Sun and panel",[["hour","Local solar time",0,24,.1,10,"h"],["tilt","Panel tilt \u03B2",0,90,1,30,"\xB0"],["azimuth","Panel azimuth \u03B3<sub>p</sub>",0,360,1,180,"\xB0"],["latitude","Latitude",-66,66,1,35,"\xB0"],["declination","Solar declination",-23.4,23.4,.1,0,"\xB0"]]],["Irradiance and temperature",[["dni","Beam irradiance scale",0,1100,10,850,"W/m\xB2"],["dhi","Diffuse irradiance scale",0,300,5,110,"W/m\xB2"],["albedo","Ground reflectance \u03C1<sub>g</sub>",0,1,.01,.2,""],["ambient","Air temperature T<sub>a</sub>",-20,55,1,25,"\xB0C"],["noct","Nominal cell temperature",25,65,1,45,"\xB0C"],["coefficient","Power temperature coefficient \u03B3<sub>P</sub>",-.7,0,.01,-.4,"%/\xB0C"]]],["Module and inverter",[["area","Module area A",.5,4,.1,2,"m\xB2"],["efficiency","Module efficiency at STC",5,30,.5,21,"%"],["inverterEfficiency","Inverter efficiency \u03B7<sub>inv</sub>",50,100,1,97,"%"],["rating","Inverter AC rating P<sub>AC,r</sub>",50,1200,10,350,"W"],["soiling","Soiling loss",0,35,1,0,"%"],["wiring","Wiring loss",0,20,1,0,"%"],["interval","Interval at current conditions \u0394t",.25,6,.25,.5,"h"]]],["I\u2013V and fill factor example",[["voc","Open-circuit voltage V<sub>oc</sub>",15,70,1,48,"V"],["isc","Short-circuit current I<sub>sc</sub>",1,20,.1,11.5,"A"],["vmp","Voltage at maximum power / V<sub>oc</sub>",45,95,1,85,"%"],["imp","Current at maximum power / I<sub>sc</sub>",45,99,1,90,"%"]]],["Solar thermal and storage example",[["flow","Fluid mass flow",0,.5,.01,.04,"kg/s"],["cp","Fluid specific heat",1e3,4500,10,4180,"J/(kg\xB7K)"],["rise","Collector outlet temperature rise",0,20,.1,3,"K"],["mass","Tank mass",25,500,5,200,"kg"],["tankRise","Tank temperature rise",0,80,1,40,"K"]]],["Battery and self-consumption example",[["load","Constant electrical load",0,600,10,100,"W"],["storage","Usable battery capacity",0,3e3,50,500,"Wh"],["batteryPower","Charge and discharge power limit",0,600,10,200,"W"],["batteryEfficiency","Efficiency per direction",50,100,1,90,"%"]]],["Degradation and lifetime cost example",[["firstYear","First-year AC energy (independent estimate)",50,1500,10,500,"kWh"],["degradation","Annual degradation",0,5,.1,.5,"%"],["years","Operating life",1,40,1,25,"years"],["capital","Installed cost",0,5e3,25,500,"\u20AC"],["maintenance","Annual operating cost",0,250,5,10,"\u20AC"],["discount","Real discount rate",0,15,.1,3,"%"]]],["Availability and performance checks",[["available","Equipment available time",0,24,.25,22,"h"],["calibration","Training observed / expected energy",.5,1.5,.01,.9,""],["observed","Observed energy for this day",0,30,.01,2,"kWh"],["threshold","Underperformance threshold",50,100,1,80,"%"]]]],zd=Kr.flatMap(([,i])=>i),j_=new Set(Kr.slice(0,3).flatMap(([,i])=>i.map(([t])=>t))),jr=Object.fromEntries(zd.map(([i,,,,,t])=>[i,t])),Ht=(i,t=2)=>i.toFixed(t);function qc(i){return i.map(([t,e,n,s,r,a,o])=>`<div class="control"><div class="control-line"><label for="${t}">${e}</label>
    <output id="${t}-value" for="${t}" aria-hidden="true"></output></div>
    <input type="range" id="${t}" min="${n}" max="${s}" step="${r}" value="${a}"></div>`).join("")}function Q_(i,t,e,n){let l=Math.max(100,n,...t.samples.map(m=>m.dc)),u=m=>53+467*m/24,h=m=>162-146*m/l,d=m=>t.samples.map((y,g)=>`${u((g+.5)/12)},${h(y[m])}`).join(" "),f=[0,6,12,18,24].map(m=>`<text x="${u(m)}" y="185" text-anchor="middle">${m}:00</text>`).join("");i.setAttribute("viewBox","0 0 540 199"),i.innerHTML=`<path class="axis" fill="none" d="M53 16V162H520"/>
    <text x="6" y="22">${Ht(l,0)} W</text><text x="15" y="165">0 W</text>
    <polyline fill="none" stroke="#6688a0" stroke-width="2" stroke-dasharray="5 4" points="${d("dc")}"/>
    <polyline fill="none" stroke="#087660" stroke-width="3" points="${d("ac")}"/>
    <path stroke="#9a5f16" stroke-width="2" d="M${u(e)} 16V162"/>${f}`}function ty(i){let t=i.shadowRoot??i.attachShadow({mode:"open"}),e=document.createElement("style");e.textContent=rl+al;let n=document.createElement("div");n.innerHTML=`<article class="hydro-lab solar-lab" aria-labelledby="solar-title">
    <header class="lab-head"><span class="lab-kicker">Interactive engineering diagram</span>
      <h2 id="solar-title">From sunlight to electrical energy</h2>
      <p>Move through a solar day, aim the panel, and see how much electricity one module delivers.</p>
    </header>
    <div class="solar-control-panel">
      <fieldset><legend>Sun and panel</legend><div class="solar-controls">${qc(Kr[0][1])}</div></fieldset>
      ${Kr.slice(1,3).map(([h,d])=>`<details><summary>${h}</summary><div class="solar-controls">${qc(d)}</div></details>`).join("")}
    </div>
    <div class="solar-stage">
      <figure class="engineering-card"><div id="solar-scene" aria-label="Isometric solar module, sun path, inverter and AC output"></div>
        <button type="button" id="solar-play" class="motion-toggle"></button>
        <figcaption>Drag to orbit. The dashed arc shows the daylight path. Gold rays mark direct light on the panel and gold pulses trace electrical output through the inverter.</figcaption>
      </figure>
      <section class="metrics-card" aria-label="Module output">
        <div class="power-card"><span class="eyebrow">Estimated AC energy in one day</span><strong id="daily"></strong>
          <span id="status" class="status"></span></div>
        <div class="metric-list" id="metrics"></div>
      </section>
    </div>
    <div class="solar-analysis">
      <section class="chart-card"><h3>Power over the solar day</h3><p>Solid green is delivered AC, dashed blue is module DC, and the vertical marker indicates the selected time.</p>
        <svg id="solar-chart" role="img" aria-label="Daily DC and AC power in watts"></svg></section>
      <details open><summary>Live PV equations</summary><div id="equations"></div></details>
      ${Kr.slice(3).map(([h,d],f)=>`<details><summary>${h}</summary>
        <div class="solar-controls">${qc(d)}</div><div id="extension-${f}"></div></details>`).join("")}
      <p class="assumptions">Synthetic day at the selected latitude and solar declination. Azimuth is clockwise from north (east 90\xB0, south 180\xB0). Irradiance follows the sun above the horizon. Daily integration uses five-minute intervals and fixed weather and equipment settings. The selected interval assumes steady power at the current slider position.</p>
    </div>
  </article>`,t.replaceChildren(e,n);let s=h=>t.getElementById(h),r={...jr},a;try{a=Bd(s("solar-scene"))}catch{s("solar-scene").textContent="3D view unavailable. The sliders, equations and power chart still work."}let o=Wc(r),c,l=h=>`<div class="equation">${h}</div>`;function u(){let h=Jr(r),d=Math.round(r.hour*60);s("hour-value").textContent=`${String(Math.floor(d/60)).padStart(2,"0")}:${String(d%60).padStart(2,"0")}`,s("hour").setAttribute("aria-valuetext",s("hour-value").textContent+" local solar time"),s("daily").textContent=`${Ht(o.energy,3)} kWh`;let f=s("status");f.className=`status ${h.clip>0?"clipped":"open"}`,f.textContent=h.elevation<=0?"Sun below horizon":h.clip>0?"Inverter clipping":"Within inverter rating",s("metrics").innerHTML=[["AC power now",`${Ht(h.ac,1)} W`],["Module DC power now",`${Ht(h.dc,1)} W`],["Energy at current conditions",`${Ht(h.ac*r.interval/1e3,3)} kWh`],["On-panel irradiance",`${Ht(h.poa,1)} W/m\xB2`],["Sun elevation / cell temperature",`${Ht(h.elevation,1)}\xB0 / ${Ht(h.cell,1)} \xB0C`],["Daily clipping loss",`${Ht(o.clipping,3)} kWh`]].map(([x,M])=>`<div class="metric"><span>${x}</span><strong>${M}</strong></div>`).join("");let m=(Math.atan2(h.sun[0],h.sun[2])/en+360)%360;s("equations").innerHTML=[`\u03B8<sub>z</sub> = ${Ht(90-h.elevation,1)}\xB0 \xB7 \u03B3<sub>s</sub> = ${Ht(m,1)}\xB0 \xB7 cos \u03B8<sub>i</sub> = cos \u03B8<sub>z</sub> cos \u03B2 + sin \u03B8<sub>z</sub> sin \u03B2 cos(\u03B3<sub>s</sub> \u2212 \u03B3<sub>p</sub>) = ${Ht(h.incidence,3)}`,`GHI = DNI \xD7 max(0, cos \u03B8<sub>z</sub>) + DHI = ${Ht(h.dni,1)} \xD7 ${Ht(Math.max(0,h.sun[1]),3)} + ${Ht(h.dhi,1)} = ${Ht(h.ghi,1)} W/m\xB2`,`G<sub>POA</sub> = DNI \xD7 max(cos \u03B8<sub>i</sub>, 0) + DHI \xD7 (1 + cos \u03B2)/2 + \u03C1<sub>g</sub> \xD7 GHI \xD7 (1 \u2212 cos \u03B2)/2 = ${Ht(h.beam,1)} + ${Ht(h.sky,1)} + ${Ht(h.ground,1)} \u2248 ${Ht(h.poa,1)} W/m\xB2`,`P<sub>STC</sub> = G<sub>STC</sub> \xD7 A \xD7 \u03B7<sub>STC</sub> = ${ol} \xD7 ${r.area} \xD7 ${r.efficiency/100} = ${Ht(h.rated,1)} W \xB7 P<sub>DC,constant \u03B7</sub> = G<sub>POA</sub> \xD7 A \xD7 \u03B7<sub>STC</sub> = ${Ht(h.areaPower,1)} W`,`T<sub>c</sub> = T<sub>a</sub> + (G<sub>POA</sub>/800)(NOCT \u2212 20) = ${Ht(h.cell,1)} \xB0C`,`P<sub>DC</sub> = P<sub>STC</sub> \xD7 (G<sub>POA</sub>/G<sub>STC</sub>) \xD7 max[0, 1 + \u03B3<sub>P</sub>(T<sub>c</sub> \u2212 25)] = ${Ht(h.areaPower,1)} \xD7 ${Ht(h.tempFactor,3)} \u2248 ${Ht(h.dc,1)} W \xB7 \u03B7<sub>PV</sub> = P<sub>DC</sub>/(G<sub>POA</sub>A) = ${h.poa>0?Ht(100*h.dc/(h.poa*r.area),1)+"%":"undefined in darkness"}`,`Retained fraction = \u220F(1 \u2212 \u2113<sub>j</sub>) = ${Ht(h.retained,3)} \xB7 P<sub>AC,pot</sub> = \u03B7<sub>inv</sub> \xD7 P<sub>DC</sub> \xD7 retained fraction = ${Ht(h.potential,1)} W \xB7 inverter conversion loss = ${Ht(h.conversionLoss,1)} W`,`P<sub>AC</sub> = min(P<sub>AC,pot</sub>, P<sub>AC,r</sub>) = ${Ht(h.ac,1)} W \xB7 P<sub>clip</sub> = max(P<sub>AC,pot</sub> \u2212 P<sub>AC,r</sub>, 0) = ${Ht(h.clip,1)} W \xB7 DC/AC rating ratio = ${Ht(h.rated/r.rating)}`,`E<sub>interval</sub> = P<sub>AC</sub> \xD7 \u0394t / 1000 = ${Ht(h.ac,1)} \xD7 ${r.interval} / 1000 \u2248 ${Ht(h.ac*r.interval/1e3,3)} kWh`,`E<sub>AC,day</sub> = \u03A3 P<sub>AC,i</sub> \xD7 \u0394t<sub>i</sub> / 1000 = ${Ht(o.energy,3)} kWh \xB7 H<sub>POA</sub> = \u03A3 G<sub>POA,i</sub> \xD7 \u0394t<sub>i</sub> / 1000 = ${Ht(o.irradiation,3)} kWh/m\xB2`,`Y<sub>f</sub> = E<sub>AC</sub>/P<sub>STC</sub> = ${Ht(o.yieldFinal)} kWh/kWp \xB7 Y<sub>r</sub> = H<sub>POA</sub>/G<sub>STC</sub> = ${Ht(o.yieldRef)} h \xB7 PR = Y<sub>f</sub>/Y<sub>r</sub> = ${o.pr===null?"undefined":Ht(o.pr,3)} \xB7 AC capacity factor = E<sub>AC</sub>/(P<sub>AC,r</sub> \xD7 24 h) = ${Ht(o.cf*100,1)}% (ratings in kW)`].map(l).join("");let y=r.voc*r.vmp/100,g=r.isc*r.imp/100;s("extension-0").innerHTML=l(`Independent datasheet example. P<sub>mp</sub> = V<sub>mp</sub>I<sub>mp</sub> = ${Ht(y)} V \xD7 ${Ht(g)} A \u2248 ${Ht(y*g)} W \xB7 FF = P<sub>mp</sub>/(V<sub>oc</sub>I<sub>sc</sub>) = ${Ht(y*g/(r.voc*r.isc),3)}`);let p=r.flow*r.cp*r.rise,v=h.poa>0?p/(h.poa*r.area):null;s("extension-1").innerHTML=l(`Q\u0307<sub>u</sub> = \u1E41c<sub>p</sub>(T<sub>out</sub> \u2212 T<sub>in</sub>) = ${Ht(p,1)} W of heat \xB7 \u03B7<sub>th</sub> = Q\u0307<sub>u</sub>/(G<sub>POA</sub>A) = ${v===null?"undefined without sunlight":Ht(v*100,1)+"%"}`)+l(`Q<sub>stored</sub> \u2248 mc<sub>p</sub>\u0394T = ${Ht(r.mass*r.cp*r.tankRise/36e5,3)} kWh of heat`)+(p>h.poa*r.area?"<p>These thermal inputs require more heat than the incident sunlight supplies.</p>":"");let b=Ud(o.samples,r);s("extension-2").innerHTML=l("Starts empty. PV serves the load first, then charges the battery. \u0394SOC = (\u03B7 \xD7 P<sub>charge</sub> \u2212 P<sub>discharge</sub>/\u03B7) \xD7 \u0394t.")+l(`Load served: ${Ht(b.used,3)} kWh \xB7 Unmet load: ${Ht(b.unmet,3)} kWh \xB7 Surplus PV: ${Ht(b.surplus,3)} kWh \xB7 Final stored energy: ${Ht(b.stored,3)} kWh`);let _=Fd(r);s("extension-3").innerHTML=l(`E<sub>y</sub> = E<sub>1</sub>(1 \u2212 d)<sup>y\u22121</sup> \xB7 Final year: ${Ht(_.lastYear,1)} kWh \xB7 Lifetime: ${Ht(_.energy,1)} kWh`)+l(`LCOE = [C<sub>0</sub> + \u03A3 C<sub>y</sub>/(1 + r)<sup>y</sup>] / [\u03A3 E<sub>y</sub>/(1 + r)<sup>y</sup>] = \u20AC${Ht(_.lcoe,3)}/kWh`);let T=o.energy*r.calibration,w=T>0?r.observed/T:null;s("extension-4").innerHTML=l(`Time availability = available hours / 24 h = ${Ht(r.available/24*100,1)}%. This standalone indicator does not change PV generation.`)+l(`Calibrated prediction = modeled daily AC energy \xD7 training ratio = ${Ht(T,3)} kWh. Observed / predicted = ${w===null?"undefined at zero prediction":Ht(w*100,1)+"%"}.`)+l(w===null?"No performance flag without a positive prediction.":w<r.threshold/100?"Flagged below the selected threshold.":"At or above the selected threshold."),Q_(s("solar-chart"),o,r.hour,r.rating),a?.update(r,h,!!c)}for(let[h,,,,,,d]of zd){let f=s(h),m=()=>{r[h]=Number(f.value),h!=="hour"&&(s(`${h}-value`).textContent=`${f.value} ${d}`.trim(),f.setAttribute("aria-valuetext",s(`${h}-value`).textContent))};m(),f.addEventListener("input",()=>{m(),h!=="hour"&&j_.has(h)&&(o=Wc(r)),u()})}return ks(s("solar-play"),!1,"solar day"),s("solar-play").onclick=()=>{c?(clearInterval(c),c=void 0):c=setInterval(()=>{document.hidden||!i.isConnected||(s("hour").value=String(Math.round(r.hour*10+1)%240/10),r.hour=Number(s("hour").value),u())},110),ks(s("solar-play"),!!c,"solar day"),u()},u(),{destroy(){clearInterval(c),a?.destroy(),t.replaceChildren()}}}var kd=document.getElementById("solar-lab-root");kd&&ty(kd);var ey=new Vt("#b1efff"),ny=new Vt("#ffd097"),iy=new Vt("#75cde7"),sy=new Vt("#e0a16f");function Gd(i,t,e,{fractured:n=!1}={}){let s=new ce,r=new ce,a=new ce,o=new ce;o.visible=!1,s.add(r,a,o),i.add(s);let[c,l,u]=t,[h,d]=e,f=(b,_)=>new F(oi.lerp(c,h,b),oi.lerp(l,d,b)+_,u),m=new De(.065,10,8),y=new Vt;function g(b,_,T){let w=new de().setFromPoints(b),x=b.flatMap(C=>y.copy(iy).lerp(sy,oi.clamp((C.x-c)/(h-c),0,1)).toArray());w.setAttribute("color",new Jt(x,3));let M=new Hi({color:16777215,vertexColors:!0,transparent:!0,opacity:T,depthWrite:!1,dashSize:n?.26:.12,gapSize:.1}),P=new Ss(w,M);P.computeLineDistances(),_.add(P)}let v=(n?[[[0,0],[.15,-.12],[.3,.18],[.42,-.2],[.55,.18],[.72,-.28],[.88,-.09],[1,0]],[[0,0],[.13,.23],[.27,.42],[.42,-.2],[.59,-.38],[.73,-.4],[.88,-.09],[1,0]],[[0,0],[.15,-.12],[.31,-.48],[.44,-.44],[.55,.18],[.71,.34],[.86,.27],[1,0]]]:[-.33,0,.4].map(b=>[[0,0],[.18,b*.55],[.4,b],[.62,b*.8],[.84,b*.6],[1,0]])).map(b=>{let _=b.map(([x,M])=>f(x,M)),T=n?new Es:new tn(_,!1,"centripetal");if(n){for(let x=1;x<_.length;x++)T.add(new As(_[x-1],_[x]));g(_.slice(0,4),o,.86),g(_.slice(5),o,.86)}g(T.getPoints(90),r,n?.86:.52);let w=Array.from({length:n?6:8},()=>{let x=new he(m,new Ae);return r.add(x),x});return{curve:T,beads:w,distance:T.getLength()}});if(n)for(let b of[[[.13,.23],[.14,.43],[.21,.49]],[[.27,.42],[.25,.58],[.36,.62]],[[.31,-.48],[.28,-.58],[.39,-.58]],[[.71,.34],[.65,.47],[.72,.55]],[[.86,.27],[.87,.43],[.92,.48]],[[.73,-.4],[.78,-.54],[.88,-.56]]])g(b.map(([_,T])=>f(_,T)),a,.65);return{setConnected(b){r.visible=b,o.visible=!b},draw(b,_){v.forEach(({curve:T,beads:w,distance:x},M)=>{w.forEach((P,C)=>{if(P.visible=_,!_)return;let k=(b/x+C/w.length+M*.13)%1;P.position.copy(T.getPointAt(k)),P.position.z+=.035,P.material.color.copy(ey).lerp(ny,k)})})}}}function Vd(i){let t=(U,et=0)=>new Re({color:U,roughness:.58,metalness:et}),e=t("#8396a1",.48),n=t("#344954",.45),s=t("#cbd1ca"),r=t("#b7c4c8",.4),a=t("#b86c47",.38),o=t("#c45634"),c=t("#247fae"),l=t("#8858ac"),u=t("#477f95"),h=[],d=[];function f(U,et,V,N,L=i){let I=new he(U,et);return I.position.set(...V),I.castShadow=I.receiveShadow=!0,N&&(I.userData.label=N,h.push(I)),L.add(I),I}let m=(U,et,V,N,L)=>f(new Ce(...U),et,V,N,L),y=(U,et,V,N,L,I,G)=>f(new Le(U,et,V,28),N,L,I,G);function g(U,et,V,N){let L=new F(...U),I=new F(...et).sub(L);y(V,V,I.length(),n,L.addScaledVector(I,.5).toArray(),N).quaternion.setFromUnitVectors(new F(0,1,0),I.normalize())}let p=new De(.085,12,8);function v(U,et,V,N,L,I,G=12){let ht=new tn(U.map($=>new F(...$)),!1,"centripetal");f(new un(ht,64,et,12,!1),V,[0,0,0],I);let ut=new Ae({color:N}),E=Array.from({length:G},()=>{let $=f(p,ut,[0,0,0]);return $.castShadow=!1,$});d.push({curve:ht,pulses:E,channel:L,radius:et,phase:0,distance:ht.getLength()})}m([16.8,.3,7.2],t("#738765"),[0,-.13,-.55],"Ground surface"),m([16.7,.65,7.1],t("#a48669"),[0,-.6,-.55],"Overburden"),m([16.65,2.25,7.03],t("#8d8073"),[0,-2.03,-.55],"Cap rock"),m([16.6,1.34,6.98],t("#587f8c"),[0,-3.79,-.55],"Water-bearing geothermal reservoir"),m([16.55,.62,6.93],t("#675b60"),[0,-4.75,-.55],"Hot basement rock");let b=t("#ab7359");m([16.52,.18,6.9],b,[0,-4.64,-.55],"Hot rock beneath reservoir");let _=new xr(.19,0),T=[t("#756d65"),t("#aa9984"),t("#71969c"),t("#426875")],w=6123,x=()=>(w=Math.imul(w,1664525)+1013904223>>>0,w/2**32);for(let U=0;U<46;U++){let et=-7.9+x()*15.8,V=U%2?-3.26-x()*1.04:-1.01-x()*1.94,N=f(_,T[U%4],[et,V,2.96],"");N.scale.set(.55+x()*.8,.36+x()*.6,.37),N.rotation.z=x()*2*Math.PI}let M=[t("#4d725c"),t("#638067")],P=t("#665447"),C=new Tn(.44,.94,10);for(let[U,et,V]of[[-7.1,-2.85,1.15],[-6,-2.8,.9],[-4.95,-3.35,.85],[-3.85,-3.2,1.12],[.9,-3.25,.85],[2,-3.4,1.15],[7.3,-3.3,.85]]){y(.075,.11,.81*V,P,[U,.4*V,et],"Tree");for(let N=0;N<3;N++)f(C,M[N%2],[U,(.94+.3*N)*V,et],"Tree").scale.setScalar(V*(1-N*.13))}m([11,.14,3.15],s,[-.7,.08,.8],"Plant service pad");for(let[U,et]of[[-5.9,"Production wellhead"],[5.45,"Reinjection wellhead"]])y(.45,.52,.28,s,[U,.16,2.56],et),y(.23,.23,.53,e,[U,.41,2.56],et),y(.37,.37,.11,r,[U,.62,2.56],et);m([1.52,1.32,1.26],n,[-3.5,1.03,.9],"Heat exchanger keeps the two fluids separate");for(let U=0;U<9;U++)m([1.62,.049,1.35],r,[-3.5,.44+U*.141,.9],"Heat-exchanger plates");let O=[-.85,1.05,.74],j=[1.22,1.05,.74];m([4.1,.29,1.72],s,[.1,.25,.74],"Turbine-generator foundation"),m([4.15,1.7,.17],t("#c0cecc"),[.11,1,-.14],"Open powerhouse wall"),m([4.25,.16,.54],s,[.11,1.9,.04],"Powerhouse roof cutaway");let D=y(.135,.135,3.08,e,[.15,1.05,.74],"Turbine\u2013generator shaft");D.rotation.z=Math.PI/2;let K=new Le(.65,.65,.63,32,1,!0,Math.PI/2,Math.PI),Q=n.clone();Q.side=Ee;let Y=f(K,Q,[-1.05,1.05,.74],"Turbine casing");Y.rotation.z=Math.PI/2;let J=new ce;J.position.set(-1.05,1.05,.74),i.add(J);let at=y(.19,.19,.19,r,[0,0,0],"Turbine runner",J);at.rotation.z=Math.PI/2;for(let U=0;U<9;U++){let et=U*Math.PI*2/9;m([.12,.33,.12],U===0?a:e,[0,.35*Math.cos(et),.35*Math.sin(et)],"Turbine blade",J).rotation.x=et+.37}let W=y(.52,.52,1.12,a,j,"Generator");W.rotation.z=Math.PI/2;for(let U=0;U<8;U++){let et=y(.55,.55,.045,e,[.71+U*.144,1.05,.74],"Generator cooling fins");et.rotation.z=Math.PI/2}m([1.5,.75,1.18],u,[3.4,.48,1.23],"Condenser"),m([1.67,1.14,1.12],e,[3.4,.76,-.4],"Dry cooler rejects cycle heat");for(let U=0;U<9;U++)m([.05,.58,.06],n,[2.77+U*.16,.83,.18],"Dry cooler fins");for(let U of[3.02,3.78])y(.25,.25,.07,n,[U,1.37,-.39],"Dry cooler fan");y(.26,.26,.43,e,[-2.24,.52,1.95],"Working-fluid circulation pump"),m([1.16,.9,.84],n,[1.55,.5,-1.32],"Step-up transformer");for(let U=0;U<7;U++)m([1.26,.052,.94],e,[1.55,.24+.105*U,-1.32],"Transformer fins");let tt=[5.9,-1.9];for(let U of[-1,1]){for(let et of[-1,1])g([tt[0]+U*.57,.02,tt[1]+et*.45],[tt[0]+U*.14,4.65,tt[1]+et*.12],.036,"Grid transmission tower");for(let et=0;et<5;et++)g([tt[0]-U*(.58-et*.083),et*.83,tt[1]+.39],[tt[0]+U*(.5-et*.083),(et+1)*.83,tt[1]+.33],.026,"Tower bracing")}g([4.82,4,tt[1]],[6.98,4,tt[1]],.045,"Grid transmission crossarm");for(let U of[-.84,0,.84])y(.055,.055,.23,r,[1.55+U*.45,1.03,-1.32],"Transformer bushing"),y(.055,.055,.3,r,[tt[0]+U,3.86,tt[1]],"Transmission insulator"),v([[1.55+U*.45,1.14,-1.32],[3.72+U*.45,2.1,-1.55],[5.9+U,3.71,-1.9],[7+U*.45,3.27,-2.58],[8.25+U*.24,3.58,-3.16]],.029,n,"#ffd266","grid","Grid transmission conductor",6);v([[1.69,1.2,.72],[1.8,.62,.26],[1.55,.48,-.43],[1.55,.65,-1.12]],.052,n,"#ffd266","gross","Generator output to transformer",5),v([[-5.1,-3.8,2.99],[-5.86,-3.76,3.04],[-5.9,-2.95,3.03],[-5.9,.47,3.03],[-5.9,.78,2.57],[-5.25,.78,2.42],[-4.35,.78,1.2],[-3.99,.78,.9]],.145,o,"#ffab60","brine","Hot brine rises through the production well"),v([[-3.93,.63,1.18],[-3.38,.42,2.55],[-2.3,.39,2.65],[3.95,.39,2.65],[5.45,.64,2.57],[5.45,.46,3.03],[5.45,-3.47,3.03],[5.28,-3.88,3.04],[4.58,-3.88,3.04]],.118,c,"#8ee9ff","brine","Cooled brine descends into the reinjection well"),v([[-3.02,1.45,.7],[-2.55,1.53,.52],[-1.54,1.4,.55],O],.12,l,"#ecc2ff","cycle","Organic vapor from heat exchanger to turbine",7),v([[-.85,1.04,.48],[-.56,.59,.13],[.3,.53,-.46],[2.15,.44,.05],[2.93,.54,1.11]],.11,l,"#ecc2ff","cycle","Turbine exhaust to condenser",8),v([[3.2,.4,1.49],[2.88,.33,1.97],[.94,.33,1.97],[-2.24,.52,1.95],[-3.32,.78,1.58],[-3.5,.88,1.31]],.105,u,"#a6ddf4","cycle","Condensed working fluid returns through the circulation pump",10),v([[3.49,.43,.82],[3.49,.49,.15],[3.49,1.11,-.46]],.086,u,"#a6ddf4","cycle","Condenser coolant to dry cooler",5),v([[3.81,1.11,-.46],[3.81,.52,.11],[3.81,.48,.79]],.086,c,"#8ee9ff","cycle","Coolant return from dry cooler",5);let vt=Gd(i,[4.58,-3.88,3.08],[-5.1,-3.8,3.08]),wt=0;function st(U,et,V){V&&(J.rotation.x=(J.rotation.x+U*Math.min(7,et.gross*1.5))%(Math.PI*2));for(let N of d){let L=et[N.channel];V&&(N.phase=(N.phase+U*Math.min(4,.5+L)/N.distance)%1),N.pulses.forEach((I,G)=>{I.visible=L>0,I.visible&&(I.position.copy(N.curve.getPointAt((G/N.pulses.length+N.phase)%1)),I.position.z+=N.radius*.85)})}V&&et.brine>0&&(wt+=U*Math.min(4,.5+et.brine)),vt.draw(wt,et.brine>0)}return{advance:st,clickable:h,routes:d,runner:J,warmRock:b,reservoirFlow:vt}}var Yc={reservoir:150,wellLoss:0,reinjection:70,flow:50,cp:4180,depth:3,sink:25,efficiency:12,parasitic:10,pressure:2,density:1e3,pumpEfficiency:75,auxiliaries:.05,rating:5,hours:4,availability:95,decline:2,years:25,surface:15,gradient:35,conductivity:2.5,permeability:50,viscosity:.5,pressureGradient:10,volume:5,porosity:12,rockDensity:2600,rockCp:900,heatPumpPower:.5,cop:3.5,loadMode:"fraction"};function Hd(i){let t=i.reservoir-i.wellLoss,e=Math.min(t,Math.max(i.sink,i.reinjection)),n=t-e,s=t+273.15,r=e+273.15,a=i.sink+273.15,o=Math.max(0,1-a/s),c=n>0?Math.max(0,1-a*Math.log1p(n/r)/n):0,l=Math.min(i.efficiency/100,c),u=i.cp*n,h=i.flow*u/1e6,d=h*l,f=i.flow/i.density,m=i.pressure*f/(i.pumpEfficiency/100),y=i.loadMode==="explicit"?m+(i.flow>0?i.auxiliaries:0):d*i.parasitic/100,g=d-y,p=Math.min(i.rating,Math.max(g,0)),v=Math.max(0,-g),b=Array.from({length:i.years},(P,C)=>p*(1-i.decline/100)**C*8760*i.availability/100),_=i.surface+i.depth*i.gradient,T=i.conductivity*i.gradient,w=i.permeability*9869233e-22/(i.viscosity/1e3)*i.pressureGradient*1e3,x=i.porosity/100,M=i.volume*1e6*((1-x)*i.rockDensity*i.rockCp+x*i.density*i.cp)*(i.reservoir-i.surface)/36e11;return{production:t,reinjection:e,delta:n,carnot:o,streamLimit:c,conversion:l,enthalpy:u,thermal:h,gross:d,volumeFlow:f,pump:m,parasitic:y,net:g,exported:p,imported:v,rejected:h-d,curtailed:Math.max(0,g-p),interval:p*i.hours,annual:b,lifetime:b.reduce((P,C)=>P+C,0),cf:b[0]/(i.rating*8760),depthTemp:_,heatFlux:T,darcy:w,stored:M}}function Wd(i,t){let e,n,s,r,a;switch(i){case"hydro":e=new tl(t),e.build(),a=["Gross head",5,250,Ei.grossHeadM,"m"],s=l=>(n=Ad({...Ei,grossHeadM:l}),e.update(n),n.plantMW.toFixed(2)+" MW"),r=l=>e.advance(l);break;case"wind":e=Gc(t),a=["Wind speed",0,30,8,"m/s"],s=l=>{let u={...Hc,speed:l,heightMode:"hub",outputMode:"curve"};return n=sl(u),e.head.position.set(e.towerX,7.85,0),e.head.rotation.y=u.direction*en,e.air.position.copy(e.head.position),e.air.rotation.y=e.head.rotation.y,n.gridMW.toFixed(2)+" MW"},r=l=>e.advance(l,n,!0);break;case"solar":e=Xc(t),a=["Local solar time",0,24,10,"h"],s=l=>{n=Jr(jr,l),e.rack.rotation.y=jr.azimuth*en,e.panel.rotation.x=jr.tilt*en,e.sun.position.copy(e.sunPoint(n.sun)),e.sun.visible=n.elevation>0,e.led.color.set(n.dc>0?"#67db9f":"#425a63"),e.output.color.set(n.ac>0?"#ffc153":"#56686d");let u=new F(...n.sun),h=[];for(let d of[-1.15,0,1.15])for(let f of[-.7,.7]){let m=t.worldToLocal(e.panel.localToWorld(new F(d,.15,f)));h.push(m.clone().addScaledVector(u,3.2),m)}return e.rays.geometry.dispose(),e.rays.geometry=new de().setFromPoints(h),e.rays.visible=n.beam>0,n.ac.toFixed(0)+" W"},r=l=>e.advance(l,n,!0);break;case"geothermal":e=Vd(t),a=["Brine flow",0,100,Yc.flow,"kg/s"],s=l=>(n=Hd({...Yc,flow:l}),n.exported.toFixed(2)+" MW"),r=l=>e.advance(l,{brine:a[3]/30,cycle:n.thermal/12,gross:n.gross,grid:n.exported},!0);break}function o(l){return a[3]=l,s(l)}let c=o(a[3]);return{control:a,set:o,advance:r,initial:c}}function Xd(i,t){let e=731,n=()=>(e=Math.imul(e,1664525)+1013904223>>>0)/4294967296,s=f=>new Re({color:f,roughness:.92,flatShading:!0}),r=new Map,a={box:new Ce,rock:new Ar(1,0),pine:new Tn(1,1,8),post:new Le(1,1,1,9)},o=new Ne;function c(f,m,y,g,p,v,b,_,T=0){let w=f+m;r.has(w)||r.set(w,{shape:f,color:m,matrices:[]}),o.position.set(y,g,p),o.scale.set(v,b,_),o.rotation.set(0,T,0),o.updateMatrix(),r.get(w).matrices.push(o.matrix.clone())}function l(f,m,y,g=0){c("post",6703931,f,g+.65*y,m,.16*y,1.3*y,.16*y);for(let p=0;p<3;p++)c("pine",[3233870,4221781,5537885][p],f,g+(1.3+p*.7)*y,m,(1.04-p*.2)*y,(2.1-p*.28)*y,(1.04-p*.2)*y)}function u(f,m,y){let p=[],v=[],b=[[-3.6,1.17,5540496],[-1.7,1.09,13023375],[-.63,1.02,14666654],[-.22,.92,10067075],[0,.81,9676671],[0,.01,10925451]],_=Array.from({length:46},(M,P)=>1+.043*Math.sin(P*.81)+n()*.055);function T(M,P){let C=P/46*Math.PI*2;return[f+Math.sin(C)*y*_[P]*b[M][1],b[M][0],m+Math.cos(C)*y*_[P]*b[M][1]]}for(let M=0;M<b.length-1;M++)for(let P=0;P<46;P++){let C=(P+1)%46,k=T(M,P),O=T(M,C),j=T(M+1,P),D=T(M+1,C);for(let K of[[k,O,j],[O,D,j]]){let Q=new Vt(b[M][2]).multiplyScalar(.94+.1*n());for(let Y of K)p.push(...Y),v.push(Q.r,Q.g,Q.b)}}let w=new de;w.setAttribute("position",new Jt(p,3)),w.setAttribute("color",new Jt(v,3)),w.computeVertexNormals();let x=new he(w,new Re({vertexColors:!0,side:Ee,roughness:1}));i.add(x)}for(let{id:f,pos:m,radius:y}of t){u(m.x,m.z,y);for(let v=0;v<28;v++){let b=n()*Math.PI*2,_=m.x+Math.sin(b)*y*.95,T=m.z+Math.cos(b)*y*.95,w=.25+n()*.75;c("rock",v%3===0?9148048:10397338,_,-.87,T,w*1.5,w*.75,w,n()*5),v<10&&(f==="hub"||Math.cos(b)<-.25)&&l(m.x+Math.sin(b)*y*.77,m.z+Math.cos(b)*y*.77,.65+n())}if(f==="hub")continue;let g=m.length(),p=Math.atan2(m.x,m.z);for(let v=5;v<g-y*.67;v+=.56){let b=v*Math.sin(p),_=v*Math.cos(p);if(c("box",Math.floor(v*10)%3?10713168:12884848,b,-.04,_,3,.22,.53,p),Math.round((v-5)/.56)%6===0)for(let T of[-1,1]){let w=1.48*T;c("post",6837063,b+Math.cos(p)*w,-.5,_-Math.sin(p)*w,.13,3.9,.13)}}for(let v of[-1,1]){let b=g-y*.67,_=(5+b)/2;c("box",9795673,_*Math.sin(p)+v*1.48*Math.cos(p),1.08,_*Math.cos(p)-v*1.48*Math.sin(p),.12,.13,b-5,p)}}for(let f=0;f<75;f++){let m=f/75*Math.PI*2,y=145+n()*75,g=9+n()*24;c("rock",f%3?7507348:8953501,Math.sin(m)*y,-4,Math.cos(m)*y,16+n()*22,g,23,n()*6),g>26&&c("rock",12832968,Math.sin(m)*y,g*.54-4,Math.cos(m)*y,13,g*.39,14)}for(let{shape:f,color:m,matrices:y}of r.values()){let g=new dr(a[f],s(m),y.length);y.forEach((p,v)=>g.setMatrixAt(v,p)),i.add(g)}let h=new he(new ni(620,620),new dn({uniforms:{time:{value:0}},vertexShader:"varying vec3 p;void main(){p=(modelMatrix*vec4(position,1.)).xyz;gl_Position=projectionMatrix*viewMatrix*vec4(p,1.);}",fragmentShader:`varying vec3 p;uniform float time;
    void main(){vec2 q=p.xz;float t=time*.68;
      float w=sin(q.x*1.5+q.y*.51+t+sin(q.y*.7+t*.6)*1.25)*.5+.5;
      float ripple=pow(w,14.)*(.5+.5*sin(q.y*.79-q.x*.18+t*.7));
      float band=sin(q.y*.08+q.x*.03)*.5+.5;
      vec3 col=mix(vec3(.055,.34,.40),vec3(.16,.54,.55),band);
      col+=vec3(.22,.28,.24)*ripple;
      float dist=length(cameraPosition-p);col=mix(col,vec3(.69,.82,.83),smoothstep(52.,220.,dist));
      gl_FragColor=vec4(col,1.);
      #include <colorspace_fragment>
    }`.replace("#include <colorspace_fragment>",`
#include <colorspace_fragment>
`)}));h.rotation.x=-Math.PI/2,h.position.y=-1.6,i.add(h);let d=new he(new De(285,32,14),new dn({side:We,depthWrite:!1,vertexShader:"varying vec3 dir;void main(){dir=position;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.);}",fragmentShader:`varying vec3 dir;void main(){vec3 d=normalize(dir);vec3 c=mix(vec3(.77,.87,.87),vec3(.14,.43,.65),smoothstep(.01,.87,d.y));float glow=max(0.,dot(d,normalize(vec3(-.55,.46,-.66))));c+=vec3(.8,.64,.37)*(pow(glow,100.)*.45+smoothstep(.999,.9997,glow));gl_FragColor=vec4(c,1.);
#include <colorspace_fragment>
}`}));return i.add(d),f=>{h.material.uniforms.time.value=f}}var ry=new F(0,1,0),Qr=oi.clamp,ji=(i=0,t=0,e=0)=>new F(i,t,e);function ay(i){let t=new ce;i.add(t);function e(r,a,o,c,l,u=t){let h=new he(r,new Re({color:a,roughness:.83}));return h.position.set(o,c,l),u.add(h),h}e(new Bi(.3,.35,5,12),12412987,0,1.15,0),e(new De(.24,16,10),12882284,0,1.78,0),e(new Le(.255,.28,.16,16),15382868,0,1.98,0),e(new Le(.33,.33,.05,16),14263369,0,1.91,0),e(new Ce(.44,.48,.24),2443339,0,1.26,.28);let n=[-1,1].map(r=>{let a=new ce,o=new ce;return a.position.set(r*.15,.84,0),o.position.set(r*.33,1.39,0),t.add(a,o),e(new Bi(.11,.41,4,8),3493462,0,-.32,0,a),e(new Ce(.22,.16,.34),3159861,0,-.69,-.045,a),e(new Bi(.093,.31,4,8),14390101,0,-.21,0,o),{hip:a,arm:o,side:r}}),s=new he(new ki(.59,24),new Ae({color:3491901,transparent:!0,opacity:.24,depthWrite:!1}));return s.rotation.x=-Math.PI/2,i.add(s),{person:t,pose(r,a){for(let{hip:o,arm:c,side:l}of n)o.rotation.x=l*Math.sin(r)*a*.63,c.rotation.x=-o.rotation.x*.72;t.position.y=a*Math.abs(Math.sin(r))*.055+.12,s.position.set(t.position.x,.115,t.position.z)}}}function oy(i,t,e,n,s){let r=new ce;r.position.set(e,0,n),r.rotation.y=s;let a=new Re({color:6640446}),o=new he(new Ce(.17,2.3,.17),a);o.position.y=1.12,r.add(o);let c=document.createElement("canvas");c.width=512,c.height=112;let l=c.getContext("2d");l.fillStyle="#214e50",l.fillRect(0,0,512,112),l.fillStyle="#f4cf87",l.fillRect(12,12,6,88),l.font="bold 56px system-ui",l.textAlign="center",l.textBaseline="middle",l.fillStyle="#f6f3e4",l.fillText(t,264,55,462);let u=new Oi(c);u.colorSpace=He;let h=new he(new Ce(3.65,.8,.13),[a,a,a,a,new Re({map:u}),a]);h.position.y=2.18,r.add(h),i.add(r)}function qd(i,t,e,n,s){let r=new Gn({antialias:!0,powerPreference:"high-performance"});r.setPixelRatio(Math.min(devicePixelRatio,1.5)),r.toneMapping=Pn,r.toneMappingExposure=1.17;let a=r.domElement;a.setAttribute("aria-label","3D islands. Walk with WASD or the arrow keys. Drag to look around."),a.tabIndex=0,i.append(a);let o=new wn;o.fog=new or(12768983,.0082),o.add(new An(15792383,11905168,2.3));let c=new Cn(16772559,3.1);c.position.set(-36,56,31),o.add(c);let l=new Ke(52,1,.14,380),u=t.map(U=>({...U,pos:ji(U.x*4.3,0,U.z*4.3),radius:U.id==="python"?7.5:11})),h={id:"hub",name:"Energy crossing",pos:ji(),radius:6.7},d=Xd(o,[h,...u]);for(let U of u){let et=new ce;et.position.copy(U.pos),o.add(et),U.group=et;let V=U.pos.clone().normalize(),N=U.pos.clone().addScaledVector(V,-U.radius*.85);if(oy(o,U.name,N.x-V.z*2,N.z+V.x*2,Math.atan2(-V.x,-V.z)),U.id==="python"){let L=new ce;for(let G of new qr().parse(bd).paths){let ht=new Vi(qr.createShapes(G),{depth:1.15,bevelEnabled:!0,bevelSize:.15,bevelThickness:.12,bevelSegments:2});ht.translate(-12,-12,0);let ut=new he(ht,new Re({color:G.color,roughness:.55}));ut.scale.set(.15,-.15,.15),L.add(ut)}L.position.set(-2.6,2.3,-.5),et.add(L);let I=new he(new Le(1.7,2,.46,24),new Re({color:12895920}));I.position.set(-2.6,.23,-.5),et.add(I)}else et.scale.setScalar(.86),et.position.y=U.id==="geothermal"?4.55:U.id==="hydro"?1.12:.47,U.plant=Wd(U.id,et)}let{person:f,pose:m}=ay(o);f.position.copy(u[0].pos).z+=3.5;let y=new Set,g=new ft,p=null,v=null,b=[],_=null,T=.22,w=.22,x=1,M=0,P=0,C=0,k=f.position.clone(),O=!0,j=!0,D=matchMedia("(prefers-reduced-motion: reduce)").matches;function K(){let{width:U,height:et}=i.getBoundingClientRect();l.aspect=U/et,l.updateProjectionMatrix(),r.setSize(U,et,!1)}new ResizeObserver(K).observe(i),K();let Q=new Map,Y=0;a.onpointerdown=U=>{a.setPointerCapture(U.pointerId),Q.set(U.pointerId,{x:U.clientX,y:U.clientY,drag:0})},a.onpointermove=U=>{let et=Q.get(U.pointerId);if(!et)return;let V=U.clientX-et.x,N=U.clientY-et.y;if(et.drag+=Math.abs(V)+Math.abs(N),et.x=U.clientX,et.y=U.clientY,Q.size===2){let[L,I]=[...Q.values()],G=Math.hypot(L.x-I.x,L.y-I.y);Y>0&&G>0&&(x=Qr(x*Y/G,.55,1.8)),Y=G;return}T-=V*.006,w=Qr(w+N*.004,.025,1.18)},a.onpointerup=U=>{let et=Q.get(U.pointerId);if(et&&et.drag<6&&v&&v.plant){let V=a.getBoundingClientRect(),N=new ii;N.setFromCamera(new ft((U.clientX-V.left)/V.width*2-1,1-(U.clientY-V.top)/V.height*2),l);let L=N.intersectObject(v.group,!0).find(I=>I.object.userData.label);L&&s(L.object.userData.label)}Q.delete(U.pointerId),Y=0},a.onpointercancel=U=>{Q.delete(U.pointerId),Y=0},a.onwheel=U=>{U.preventDefault(),x=Qr(x*Math.exp(U.deltaY*.001),.55,1.8)};let J=["arrowup","arrowdown","arrowright","arrowleft","w","a","s","d"];addEventListener("keydown",U=>{j&&J.includes(U.key.toLowerCase())&&!v&&(y.add(U.key.toLowerCase()),U.preventDefault())}),addEventListener("keyup",U=>y.delete(U.key.toLowerCase())),addEventListener("blur",()=>{y.clear(),g.set(0,0)});let at=U=>u.reduce((et,V)=>W(U,et.pos)<W(U,V.pos)?et:V);function W(U,et){let V=Qr((U.x*et.x+U.z*et.z)/et.lengthSq(),0,1);return Math.hypot(U.x-V*et.x,U.z-V*et.z)}function tt(U){return Math.hypot(U.x,U.z)<h.radius*.84||u.some(et=>Math.hypot(U.x-et.pos.x,U.z-et.pos.z)<et.radius*.85||W(U,et.pos)<1.25)}function vt(U){U&&(v=U,b=[],_=null,T=U.id==="solar"?2.52:.25,w=.33,x=1,f.visible=!1,n(U))}function wt(){v=null,f.visible=!0,x=1,w=.24,n(null)}function st(U){v&&wt();let et=u.find(L=>L.id===U),V=at(f.position);_=et;let N=et.pos.clone().addScaledVector(et.pos.clone().normalize(),-et.radius*.71);if(b=[],V!==et){let L=Qr(f.position.dot(V.pos)/V.pos.lengthSq(),0,1);b.push(V.pos.clone().multiplyScalar(L),ji())}b.push(N)}return r.setAnimationLoop(U=>{let et=P?Math.min(.07,(U-P)/1e3):0;if(P=U,document.hidden||!j)return;M+=et;let V=f.position,N=!1;if(!v){let ut=new ft(g.x+Number(y.has("d")||y.has("arrowright"))-Number(y.has("a")||y.has("arrowleft")),g.y+Number(y.has("s")||y.has("arrowdown"))-Number(y.has("w")||y.has("arrowup"))),E=ji(),$=7;if(ut.lengthSq()>.001?(b=[],_=null,E.set(ut.x,0,ut.y).applyAxisAngle(ry,T).normalize()):b.length&&(E.copy(b[0]).sub(V),E.y=0,$=14,E.length()<$*et+.13?(V.x=b[0].x,V.z=b[0].z,b.shift(),b.length||vt(_),E.set(0,0,0)):E.normalize()),E.lengthSq()>0){let nt=V.clone().addScaledVector(E,et*$);if(tt(nt)&&(V.x=nt.x,V.z=nt.z,N=!0,C+=et*$*1.4,f.rotation.y=Math.atan2(-E.x,-E.z)),b.length&&N){let it=Math.atan2(-E.x,-E.z);T+=Math.atan2(Math.sin(it-T),Math.cos(it-T))*(1-Math.exp(-et*2.3))}}m(C,N?1:0);let B=u.find(nt=>Math.hypot(V.x-nt.pos.x,V.z-nt.pos.z)<nt.radius*.93)||null;B!==p&&(p=B,e(p))}let L=v?v.pos.clone().add(ji(0,v.id==="wind"?5:v.id==="python"?1.8:3,0)):V.clone().add(ji(0,1.7,0)),I=(v?v.id==="python"?8:Math.max(18,23/l.aspect):7.8)*x,G=ji(Math.sin(T)*Math.cos(w),Math.sin(w),Math.cos(T)*Math.cos(w)).multiplyScalar(I),ht=O||D?1:1-Math.exp(-et*6.5);k.lerp(L,ht),l.position.lerp(L.clone().add(G),ht),l.lookAt(k),O=!1;for(let ut of u)ut.plant&&(!v||v===ut)&&ut.plant.advance(D?0:et);d(D?0:M),r.render(o,l)}),{travel:st,inspect:()=>vt(p),leave:wt,stick:g,pause(U){j=!U,y.clear(),g.set(0,0)}}}var ue=i=>document.getElementById(i),$d="eage-roadmap-checkpoints-v1",Zc;try{Zc=JSON.parse(localStorage.getItem($d))}catch{}var zs=new Set(Vs.filter(i=>Array.isArray(Zc)&&Zc.includes(i.id)).map(i=>i.id)),li,Kc=null,Zd=null,ly={hydro:["Adjust head to deliver about 1 MW.",(i,t)=>t>=.95&&t<=1.05],solar:["Find an evening hour when the panel stops exporting.",(i,t)=>i>=18&&t===0],wind:["Find the wind speed that reaches 3 MW.",(i,t)=>t>=3],geothermal:["Run the plant at 30 kg/s and measure its output.",i=>i===30]},$c=new Map;function Jd(i){Zd=i,!Kc&&(ue("place").textContent=i?.name||"Energy crossing",ue("inspect").hidden=!i,ue("inspect").textContent=i?.plant?"Inspect plant":"Explore Python Island",ue("hint").textContent=i?"Walk across the bridges or choose a destination.":"Follow a bridge to an energy site.")}function cy(i){if(Kc=i,ue("console").hidden=!i,ue("drive").hidden=!!i,document.querySelector("footer").hidden=!!i,ue("mode").textContent=i?"Field station":"Island expedition",!i){Jd(Zd);return}if(ue("place").textContent=i.name,ue("hint").textContent=i.plant?"Drag to orbit \xB7 Pinch or scroll to zoom":"Open the field notes for your first Python exercise.",ue("instrument").hidden=!i.plant,!i.plant)return;let[t,e,n,s,r]=i.plant.control,a=ue("setting");Object.assign(a,{min:e,max:n,value:s}),ue("variable").textContent=t,a.oninput=()=>{ue("value").textContent=a.value+" "+r;let o=Number(a.value),c=i.plant.set(o);ue("reading").textContent=c;let[l,u]=ly[i.id];u(o,parseFloat(c))&&$c.set(i.id,o+" "+r+" \u2192 "+c),ue("mission").textContent=$c.has(i.id)?"Reading recorded \xB7 "+$c.get(i.id):l},a.oninput()}function Vn(i,t){let e=document.createElement(i);return e.textContent=t,e}function Jc(i,t){let e=Vn("a",i);return e.href=t,e}function Yd(i){return Jc(i.name+" course quiz \u2197","../../index.html#lesson="+encodeURIComponent("section7/renewableEnergyquizzes/"+i.quiz+".md"))}function Kd(i){return ue("dialog-title").textContent=i,ue("contents").replaceChildren(),ue("drawer").open||ue("drawer").showModal(),li?.pause(!0),ue("contents")}function hy(){let i=Kd("Destinations");i.append(Vn("p",zs.size===5?"Expedition complete \xB7 5 checkpoints earned":zs.size+" of 5 checkpoints earned"));for(let t of Vs){let e=Vn("button",t.name+(zs.has(t.id)?"   \u2713":"   \u2192"));e.onclick=()=>{ue("drawer").close(),li?li.travel(t.id):jd(t)},i.append(e)}}function jd(i){let t=Kd(i.name+" field notes");t.append(Vn("p",i.description),Vn("p",i.task)),i.id==="python"?t.append(Vn("pre",`power_mw = [2, 4, 3]
energy_mwh = sum(power_mw) * 1
print(energy_mwh)  # 9 MWh`),Jc("Open the Python Course \u2197","https://roderickperezaltamar.com/courses/eage-python-course.html")):t.append(Jc("Open full course lab \u2197","../"+i.id+"-lab/index.html")),t.append(Vn("h3",i.question));let e=Vn("p","");e.setAttribute("role","status");let n=i.options.map((s,r)=>{let a=Vn("button",s);return a.onclick=()=>{if(e.textContent=(r===i.correct?"Checkpoint earned. ":"Try again. ")+i.why,r===i.correct){zs.add(i.id);try{localStorage.setItem($d,JSON.stringify([...zs]))}catch{}n.forEach(o=>o.disabled=!0),zs.size===5&&t.append(Vn("h3","Expedition complete"),...Vs.map(Yd))}},a});t.append(...n,e,Yd(i))}try{li=qd(ue("world"),Vs,Jd,cy,i=>{ue("hint").textContent=i})}catch(i){console.error(i),ue("hint").textContent="3D rendering is unavailable in this browser. Open Destinations for the course labs and quizzes.",ue("drive").hidden=!0}ue("routes").onclick=hy;ue("inspect").onclick=()=>li.inspect();ue("back").onclick=()=>li.leave();ue("study").onclick=()=>jd(Kc);ue("close").onclick=()=>ue("drawer").close();ue("drawer").addEventListener("close",()=>li?.pause(!1));var Ci=ue("stick");function Qd(i){let t=Ci.getBoundingClientRect(),e=(i.clientX-t.left-t.width/2)/(t.width/2),n=(i.clientY-t.top-t.height/2)/(t.height/2),s=Math.max(1,Math.hypot(e,n));li?.stick.set(e/s,n/s),Ci.firstElementChild.style.transform="translate("+e/s*21+"px,"+n/s*21+"px)"}Ci.onpointerdown=i=>{Ci.setPointerCapture(i.pointerId),Qd(i)};Ci.onpointermove=i=>{Ci.hasPointerCapture(i.pointerId)&&Qd(i)};Ci.onlostpointercapture=()=>{li?.stick.set(0,0),Ci.firstElementChild.style.transform=""};
