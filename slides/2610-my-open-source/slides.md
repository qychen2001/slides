---
theme: seriph
title: 从薅牛毛到吃狗粮：我的开源学习之路
background: https://cover.sli.dev
class: text-center
transition: slide-left
htmlAttrs:
  lang: zh-CN
---

# 从薅牛毛到吃狗粮

<div class="mt-4 text-xl">我的开源学习之路</div>
<div class="mt-8 text-xl">陈启源</div>


<!--
大家好，我是启源，在浙大读计算机博士，同时也在做 AI 应用。先问大家一个问题：你们有没有做过一个半途而废的项目？有的举个手。好，看来不止我一个。今天我想说，半途而废不一定是坏事。接下来十五分钟，我讲三个自己的项目，讲讲我是怎么一路被问题带着走进开源的。内容不难，也不需要你会写代码，大家放松听就好。
-->

---
layout: two-cols
class: flex flex-col justify-center text-center
---

<h1 v-motion :initial="{ x: -60, opacity: 0 }" :enter="{ x: 0, opacity: 1, transition: { delay: 150, duration: 600 } }">薅牛毛</h1>

<div v-click="1" class="gloss text-3xl font-serif font-semibold">Yak Shaving</div>

::right::

<h1 v-motion :initial="{ x: 60, opacity: 0 }" :enter="{ x: 0, opacity: 1, transition: { delay: 150, duration: 600 } }">吃狗粮</h1>

<div v-click="1" class="gloss text-3xl font-serif font-semibold">Dogfooding</div>


<!--
先解释两个词。薅牛毛，英文叫 Yak Shaving：你本来想干件小事，结果得先干另一件，那件又得先干别的。比如你想做顿饭，发现锅脏了，去洗锅发现洗洁精没了，下楼买又发现电动车没电了，一个小时过去，饭还没开始做。吃狗粮，英文叫 Dogfooding，就是自己做的东西，自己先天天用。我的故事就是从薅牛毛开始的：当年为了做一份夏令营的PPT，我没先做PPT，而是先跑去做了一个模板。
-->

---
layout: image-right
image: /beamer-page01.png
backgroundSize: contain
class: flex flex-col justify-center
---

# 一个更好看的浙大 Beamer 模板

<div class="gloss text-2xl font-semibold">2023-08-22</div>

<div v-click class="mt-6 text-lg leading-relaxed">
  <a href="https://github.com/qychen2001/ZJU-Beamer-Template">github.com/qychen2001/ZJU-Beamer-Template</a>
</div>


<!--
需求很简单：要一份好看的、有浙大味道的学术PPT。找了一圈没找到满意的，那就自己做一个呗。于是有了这个仓库，ZJU Beamer Template，2023年8月22号第一次提交。做完我顺手把它公开了，传到了 Overleaf，还在 CC98 发了个帖子。然后我就去忙别的了，模板基本就搁在那儿。按一般的标准，这就是个半途而废的项目：没有持续维护，也没有什么规划。很多人做到这一步就会觉得，这事儿结束了。
-->

---
layout: two-cols
class: flex flex-col justify-center
layoutClass: gap-8
---

# PR #1

<div class="gloss text-2xl font-semibold">樊铧纬（Phil-Fan）｜2025-07-10</div>

<v-clicks>

<div class="mt-6 text-xl leading-snug">修颜色 · Makefile · 示例页</div>

<div class="mt-3 text-xl leading-snug">他的第一次开源贡献</div>

<div class="mt-4 text-lg leading-relaxed">
  <a href="https://github.com/qychen2001/ZJU-Beamer-Template/pull/1">github.com/qychen2001/ZJU-Beamer-Template/pull/1</a>
</div>

</v-clicks>

::right::

<img src="/pr1-img1.png" class="max-h-44 w-full object-contain rounded" alt="Beamer 模板封面示例" />

<img src="/pr1-img2.png" class="max-h-44 w-full object-contain rounded mt-4" alt="Beamer 模板代码示例" />


<!--
意外来了。差不多两年以后，2025年7月10号，这个仓库收到了第一个 PR。提交的是樊铧纬，GitHub 上叫 Phil-Fan。他觉得右上角的颜色不搭，看着难受，就自己动手改了，顺便加了 Makefile、参考文献、并排图片和代码的示例，还在 Mac、Ubuntu、Win11、Overleaf 上都测了一遍。这是他人生中第一次开源贡献。我自己都快忘了这个项目，它却一直在那儿等着一个人。
-->

---
layout: fact
transition: fade
---

<h1 v-motion :initial="{ y: 40, opacity: 0 }" :enter="{ y: 0, opacity: 1, transition: { delay: 150, duration: 600 } }">两年后</h1>

<div v-click class="gloss text-3xl font-bold">Agentero 贡献第一</div>


<!--
后来的故事更有意思。樊铧纬现在是我们项目 Agentero 里提交最多的贡献者，排第一。一个因为颜色不好看提的 PR，最后变成了一起做事的伙伴。从陌生人，到每天一起写代码的队友，中间就隔着一个 PR。所以价值常常是迟到的，做的时候看不到，过很久才冒出来。如果当年我觉得这个模板没什么用，没把它公开，这一切都不会发生。公开这一步几乎零成本，却给未来留了一扇门。
-->

---
layout: statement
transition: fade
---

<h1 v-motion :initial="{ y: 40, opacity: 0 }" :enter="{ y: 0, opacity: 1, transition: { delay: 150, duration: 600 } }">没有继续 ≠ 没有价值</h1>


<!--
这个模板现在一百二十多颗星，不算火，也没有一直长大。但一个项目没有继续，不代表它没有价值。我的小结是：先把眼前的小问题解决掉，然后把它放出来，让别人有机会加入。你不知道谁会路过，也不知道他会带来什么。所以下次你做了个小东西，哪怕很粗糙，也别急着藏起来。
-->

---
layout: image-right
image: /charger-web-light.png
backgroundSize: contain
class: flex flex-col justify-center
---

# 有谁用过？

<div class="gloss text-3xl font-semibold">ZJU Charger</div>

<v-clicks>

<div class="mt-6 text-lg leading-relaxed">
  <a href="https://github.com/ZJU-Charger/ZJU-Charger">github.com/ZJU-Charger/ZJU-Charger</a>
</div>

<div class="mt-3 text-lg leading-relaxed">
  <a href="https://charger.philfan.cn">charger.philfan.cn</a>
</div>

</v-clicks>


<!--
下一个故事和大家生活很近。有谁骑电动车，在学校里找过充电桩？用过 ZJU Charger 的举个手。它解决的就是：桩不好找、服务商入口不统一、状态查不方便。以前想充个电，可能要打开好几个公众号，一个个查过去。起点很朴素，就是有人受够了骑着没电的车到处找桩。2025年11月底开始做，上线没多久，网站累计访问用户就过了三千。
-->

---
layout: section
transition: fade
---

<h1 v-motion :initial="{ y: 40, opacity: 0 }" :enter="{ y: 0, opacity: 1, transition: { delay: 150, duration: 600 } }">知道哪里不对 ≠ 能改出来</h1>


<!--
做 Charger 的过程中有个现象让我印象很深。很多不写代码的同学，其实最清楚问题在哪：这个站点位置不对，那个校区少了，这里显示不准。他们知道哪里不对，却没办法把想法变成真正的改动，只能来找写代码的人。这其实是开源里很常见的困境：会写代码的人不一定最懂需求，最懂需求的人又不一定会写代码。中间那道墙，挡住了很多本来可以发生的改进。那能不能让这道墙矮一点？让知道问题的人，也能亲手把问题改掉。
-->

---
layout: image-right
image: /poco-hero.png
backgroundSize: contain
class: flex flex-col justify-center
---

# Poco

<div class="gloss text-3xl font-semibold">Your Pocket Coworker</div>

<div v-click class="mt-4 text-2xl font-medium sub">给不写代码的人</div>

<div v-click class="mt-6 text-lg leading-relaxed">
  <a href="https://github.com/poco-ai/poco-claw">github.com/poco-ai/poco-claw</a>
</div>


<!--
这就引出了 Poco。它是为不写代码的人做的。一句话讲技术：我们把写代码的 AI 智能体放到云端，套一个简单好用的聊天界面，你用大白话说想改什么，它去改，你直接看到结果。不用装环境，不用碰命令行，打开网页就能用。它的名字叫 Pocket Coworker，口袋里的同事，现在在 GitHub 上有一千三百多颗星。
-->

---
layout: center
---

<v-clicks>

<div class="text-4xl font-bold ink">学习路径从问题里长出来</div>

<div class="mt-12">
  <div class="text-4xl font-bold ink">不写代码也能贡献</div>
  <div class="text-2xl mt-3 font-medium sub">反馈 · 数据校准 · 复现 bug · 文档</div>
</div>

</v-clicks>


<!--
从找充电桩到 Poco，我想说两点。第一，学习路径不是提前规划好的，是从一个个问题里长出来的。第二，不写代码，不等于不能贡献。提反馈、校准数据、把一个 bug 稳定复现出来、补一行文档，这些都是实打实的贡献，而且往往是项目最缺的。所以在座不写代码的同学，你们完全可以是一个项目里很重要的人。
-->

---
layout: center
class: text-center
---

<div class="flex items-center justify-center gap-4 text-4xl font-bold ink">
  <span>读论文</span>
  <span class="arrow text-2xl font-normal">- - ▸</span>
  <span>问 AI</span>
  <span class="arrow text-2xl font-normal">- - ▸</span>
  <span>写汇报</span>
</div>

<div v-click class="mt-10 text-3xl font-bold punch">每一步之间都是断点</div>


<!--
第三个项目，Agentero，我用一个真实的科研场景来讲。读论文的时候，你会问 AI 很多问题：这句话什么意思，这个公式怎么推，这个实验为什么这么设计。可到了要写组会汇报，你又得重新翻论文、翻聊天记录、翻笔记，东一块西一块。明明上周问过一个特别好的问题，得到一个特别好的解释，到用的时候怎么都找不到，只能再问一遍。读、理解、整理，这三步之间全是断点。这就是我们每天都在踩的坑。我相信在座做过科研或者写过课程报告的同学，多少都有过这种感觉。
-->

---
layout: image-right
image: /agentero-hero.png
backgroundSize: contain
class: flex flex-col justify-center
---

# 为人与 Agent 协作而生的一站式科研工作台

<v-clicks>

<div class="mt-2 text-lg leading-relaxed">
  <a href="https://github.com/poco-ai/Agentero">github.com/poco-ai/Agentero</a>
</div>

<div class="mt-3 text-lg leading-relaxed">
  <a href="https://agentero.app">agentero.app</a>
</div>

</v-clicks>


<!--
那为什么不直接把现有工具拼一拼？我们试过。一个 PDF 阅读器，一个聊天 AI，一个笔记软件，来回复制粘贴。每个单独都好用，拼在一起就特别别扭：AI 不知道你在读哪一页，笔记也记不住你问过什么。所以 Agentero 想做的是：保留你读 PDF 的习惯，给模型结构化的上下文，把文献、笔记和交互记录都留在本地。读到哪儿问到哪儿，问过的东西自动沉淀下来，整理的时候直接拿来用。
-->

---
layout: image-right
image: /agentero-agent.png
backgroundSize: contain
class: flex flex-col justify-center
---

# 演示好看 ≠ 一直在用


<!--
第三层，是只有自己天天用才会发现的东西。做演示的时候，大家看的是最亮眼的那一下：一键总结、自动翻译、漂亮的界面。可一个工具能不能留下来，看的往往是那些不起眼的地方：打开快不快，上次的记录还在不在，某个按钮是不是总要多点一下。这些问题，演示的时候根本暴露不出来，只有你连着用上一周、一个月，才会一个个冒出来。而真正决定你会不会继续用下去的，恰恰就是这些小地方。所以我们自己每天都拿它读论文、写汇报，哪里别扭，第二天就改。一个东西演示起来好看，和你愿意一直用下去，完全是两回事。这也是为什么我觉得，做产品最好的老师，就是你自己的日常。
-->

---
layout: fact
transition: fade
---

<h1 v-motion :initial="{ y: 40, opacity: 0 }" :enter="{ y: 0, opacity: 1, transition: { delay: 150, duration: 600 } }">1000+ Star</h1>

<div v-click class="gloss text-2xl font-bold">2026-07 创建 · 20+ 贡献者</div>


<!--
一行数字：Agentero 今年7月创建，现在一千多颗星，二十多位贡献者。不做科研的同学也一样，课程资料、社团的活动文档、设计素材，只要你在反复地找、反复地整理，就会遇到同样的断点。别把它当成只属于科研的故事，它讲的是：当你每天被同一个小麻烦卡住，那就是一个值得动手的信号。你不需要先想清楚要做一个多大的产品，先让自己明天少翻一次文件夹，就已经很好了。
-->

---
class: flex flex-col justify-center
---

<v-clicks>

1. 从一个具体的烦恼开始，不必先学会
2. 可以浅尝，但要留下点东西
3. 先做真用户，再做改进者

</v-clicks>


<!--
最后给大家三点建议。第一，从一个具体的烦恼开始，不用等自己先学会了再动手。烦恼越具体越好，"PPT 不好看""找不到充电桩"，就够了。很多人想入门 AI 或开源，第一步是找一套完整的教程从头学，结果学了两周就放下了。不是你不够努力，而是没有一个具体的问题拉着你往前走。有了问题，你学到的每一点马上都能用上，这种感觉会推着你继续。我做模板是因为要做PPT，做 Agentero 是因为自己读论文太乱，没有一个是先学好了再开始的。
第二，浅尝辄止没关系，但请留下点东西。一个小修复、一段文档、一个能复现的问题报告都行，别只留下收藏夹里吃灰的教程。收藏不算数，留下的东西才会被人看到，就像那个模板等来了樊铧纬。哪怕你最后没坚持下去，留下的那一点点，也可能在某天帮到某个人，甚至帮到未来的你自己。很多时候，你写的那个能复现的问题报告，就是别人修好它的起点。
第三，先做一个真实的用户，再做改进者。每次做完问自己一句：明天我还会用它吗？如果自己都不想用，那它大概也只是好看。我们做 Agentero 的时候，自己每天都在用，一个按钮放得别扭，第二天自己就会觉得烦，然后就改了。这种反馈，比任何调研都来得快、来得真。
但也有个边界：你是用户，但你不是所有用户。Charger 里那些不写代码的同学，提的反馈常常是写代码的人想不到的。所以吃自己的狗粮，也要听听别人吃完怎么说。自己用，是为了诚实；听别人说，是为了不把自己的习惯当成所有人的习惯。
-->

---
layout: statement
transition: fade
---

<h1 v-motion :initial="{ y: 40, opacity: 0 }" :enter="{ y: 0, opacity: 1, transition: { delay: 150, duration: 600 } }">薅牛毛决定你往哪儿走<br>吃狗粮决定你能走多远</h1>


<!--
回到最开始。当年我只是想做一份好看的PPT，并没有先规划好整条开源路。从一个模板，到一个充电桩网站，再到 Poco 和 Agentero，每一步都是被眼前的问题带着走的。中间有很多半途而废，也有很多意外的收获。所以送大家一句话：薅牛毛决定你往哪儿走，吃狗粮决定你能走多远。
-->

---
layout: center
class: text-center
transition: fade
---

# 谢谢！

<div class="mt-8 text-xl leading-loose">
<div>GitHub：<a href="https://github.com/qychen2001">github.com/qychen2001</a></div>
<div>组织：<a href="https://github.com/poco-ai">github.com/poco-ai</a></div>
<div>邮箱：<a href="mailto:chenqiyuan1012@gmail.com">chenqiyuan1012@gmail.com</a></div>
<div>微信：17757380899</div>
</div>

<!--
这是我的联系方式，欢迎大家来 GitHub 上看看、提意见，也欢迎加我微信交流。也欢迎大家给我们的项目提反馈，哪怕只是一句“这里我看不懂”。谢谢大家！
-->
