import{d as _,z as p,o as c,b as f,e as s,f as m,h,c as v,k as g,q as k,s as $,B as l,aa as r}from"../modules/vue-BG0mrsKf.js";import{u as d,f as b}from"./context-Cj8DiaHG.js";import"../index-ChG-zILh.js";import"../modules/shiki-C89ZgNSK.js";function i(e){return e.startsWith("/")?"/talks/test1"+e.slice(1):e}function x(e,a=!1){const o=e&&["#","rgb","hsl"].some(n=>e.indexOf(n)===0),t={background:o?e:void 0,color:e&&!o?"white":void 0,backgroundImage:o?void 0:e?a?`linear-gradient(#0005, #0008), url(${i(e)})`:`url("${i(e)}")`:void 0,backgroundRepeat:"no-repeat",backgroundPosition:"center",backgroundSize:"cover"};return t.background||delete t.background,t}const y={class:"my-auto w-full"},B=_({__name:"cover",props:{background:{default:""}},setup(e){d();const a=e,o=p(()=>x(a.background,!0));return(t,n)=>(c(),f("div",{class:"slidev-layout cover",style:h(o.value)},[s("div",y,[m(t.$slots,"default")])],4))}}),C=s("h1",null,"Title",-1),S=s("p",null,[r("Hello, "),s("strong",null,"Slidev"),r("!")],-1),T=s("p",null,"This is test page 1.",-1),z=s("p",null,[r("The site is available at "),s("a",{href:"https://qychen2001.github.io/talks/test1",target:"_blank"},"HERE")],-1),P={__name:"test1.md__slidev_1",setup(e){const{$slidev:a,$nav:o,$clicksContext:t,$clicks:n,$page:w,$renderContext:E,$frontmatter:u}=d();return t.setup(),(R,q)=>(c(),v(B,k($(l(b)(l(u),0))),{default:g(()=>[C,S,T,z]),_:1},16))}},I=P;export{I as default};