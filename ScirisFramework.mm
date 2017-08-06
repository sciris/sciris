<map version="1.0.1">
<!-- To view this file, download free mind mapping software FreeMind from http://freemind.sourceforge.net -->
<node CREATED="1501095660901" ID="ID_1866433256" MODIFIED="1501095669613" TEXT="Sciris Framework">
<node CREATED="1501095710060" ID="ID_833907438" MODIFIED="1501095759545" POSITION="right" TEXT="My Objectives (7/26/17)">
<node CREATED="1501095786430" ID="ID_1294954997" MODIFIED="1501095831335" TEXT="Develop the Sciris framework for building web applications where users log into web sessions and run model simulations."/>
</node>
<node CREATED="1501095720395" ID="ID_210371767" MODIFIED="1501899567159" POSITION="right" TEXT="My Agenda (8/4/17)">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1501899509650" ID="ID_145732380" MODIFIED="1501899535690" TEXT="Submit query to StackOverflow about how to get mpld3 working with Vue and webpack.">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1501899619767" ID="ID_524731174" MODIFIED="1501899631263" TEXT="Get scatterplotter_p1 app working.">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1501297526016" ID="ID_1694767845" MODIFIED="1501297538302" TEXT="due Monday, 8/14/17"/>
<node CREATED="1501899643920" ID="ID_104761882" MODIFIED="1501899661962" TEXT="original idea was to implement my Shiny R polyfitter app"/>
</node>
<node CREATED="1501096688972" ID="ID_962056441" MODIFIED="1501096699059" TEXT="Get familiar with Vue.">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1501091624919" ID="ID_1785334982" MODIFIED="1501273890230" TEXT="Figure out how I want to structure the Sciris framework.">
<font NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1501091624920" ID="ID_1178434646" MODIFIED="1501273884492" TEXT="Figure out how I want to create a Nutrition GUI.">
<font NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1501091624920" ID="ID_110713713" MODIFIED="1501263605197" TEXT="In the long run, figure out how I want to port Optima HIV to Vue / Sciris.">
<font NAME="SansSerif" SIZE="12"/>
</node>
</node>
<node CREATED="1501177157135" ID="ID_1508619691" MODIFIED="1501177161153" POSITION="right" TEXT="My Schedule">
<node CREATED="1501177162267" ID="ID_695878909" MODIFIED="1501177190563" TEXT="The Sciris framework _and_ Nutrition sites basically need to be done by the end of the year!"/>
</node>
<node CREATED="1501096028227" ID="ID_1672136765" MODIFIED="1501096030919" POSITION="right" TEXT="My Questions"/>
<node CREATED="1501096031224" ID="ID_1471998132" MODIFIED="1501096036136" POSITION="right" TEXT="My Problems / Challenges">
<node CREATED="1501177195348" ID="ID_766254759" MODIFIED="1501177220187" TEXT="Getting both a viable framework up and a Nutrition site by the end of the year"/>
</node>
<node CREATED="1501095915336" ID="ID_786544597" MODIFIED="1501095922607" POSITION="right" TEXT="Project Notes">
<node CREATED="1501096952143" ID="ID_565156384" MODIFIED="1501096955433" TEXT="Name of the framework">
<node CREATED="1501096924924" ID="ID_1016214052" MODIFIED="1501096946065" TEXT="Sciris = scientific iris">
<node CREATED="1501096959261" ID="ID_1807178309" MODIFIED="1501096988798" TEXT="Cliff came up with this as a reference to Iris, messenger (goddess) of the gods."/>
<node CREATED="1501096989488" ID="ID_1149774932" MODIFIED="1501097049532" TEXT="It is also appropriate, maybe, in the sense that an iris is the aperture of the eye, and this tool should allow users a view on the simulation data they run in the model."/>
</node>
</node>
<node CREATED="1501097092711" ID="ID_1879825786" MODIFIED="1501214423577" TEXT="Framework Architecture and Code Notes">
<node CREATED="1501097202877" ID="ID_445912805" MODIFIED="1501097207814" TEXT="Web User Interface">
<node CREATED="1501097175052" ID="ID_719065688" MODIFIED="1501098549337" TEXT="Vue Client (JavaScript)"/>
</node>
<node CREATED="1501097224009" ID="ID_1364833291" MODIFIED="1501097228497" TEXT="Web Session Manager">
<node CREATED="1501097304857" ID="ID_1181960261" MODIFIED="1501097343885" TEXT="Database (Postgres / Redis)"/>
<node CREATED="1501097323244" ID="ID_423782657" MODIFIED="1501097350087" TEXT="Twisted / Flask Server (Python)"/>
<node CREATED="1501098448108" ID="ID_559510755" MODIFIED="1501098453883" TEXT="Celery Task Manager (Python)">
<node CREATED="1501098454710" ID="ID_4365760" MODIFIED="1501098466669" TEXT="develop / port this later"/>
</node>
</node>
<node CREATED="1501097239951" ID="ID_1442063781" MODIFIED="1501097263491" TEXT="Domain Model">
<node CREATED="1501097291606" ID="ID_1630627425" MODIFIED="1501098475700" TEXT="Domain Model Code (Python)"/>
</node>
<node CREATED="1501104224197" ID="ID_1807022988" MODIFIED="1501104224197" TEXT=""/>
<node CREATED="1501104211501" ID="ID_173406084" MODIFIED="1501104242692" TEXT="Objects Across All Parts of Framework">
<node CREATED="1501104264012" ID="ID_562163767" MODIFIED="1501104270989" TEXT="User">
<node CREATED="1501104273168" ID="ID_620871208" MODIFIED="1501104282081" TEXT="a user of the web application"/>
</node>
<node CREATED="1501104288992" ID="ID_1967193107" MODIFIED="1501104291180" TEXT="WebSession">
<node CREATED="1501104292744" ID="ID_1355678401" MODIFIED="1501104303017" TEXT="a session where a User is logged in"/>
</node>
<node CREATED="1501104315573" ID="ID_301412801" MODIFIED="1501104317556" TEXT="Project">
<node CREATED="1501104318400" ID="ID_1140449326" MODIFIED="1501104386143" TEXT="a particular exploration a User is doing, including saved data"/>
</node>
<node CREATED="1501104410345" ID="ID_1240998051" MODIFIED="1501104411648" TEXT="Graph">
<node CREATED="1501104412616" ID="ID_407547845" MODIFIED="1501104429310" TEXT="a graphical figure"/>
<node CREATED="1501104430099" ID="ID_636272877" MODIFIED="1501104441415" TEXT="probably implemented using mpld3"/>
</node>
<node CREATED="1501104830949" ID="ID_357798717" MODIFIED="1501104837817" TEXT="Optimizer">
<node CREATED="1501265261601" ID="ID_928655010" MODIFIED="1501265267596" TEXT="optimization method"/>
<node CREATED="1501265258648" ID="ID_1953791311" MODIFIED="1501265258648" TEXT="instanced by ASD algorithm optimizer"/>
</node>
<node CREATED="1501105061027" ID="ID_547854421" MODIFIED="1501105068713" TEXT="ScirisModel">
<node CREATED="1501265285108" MODIFIED="1501265285108" TEXT="superclass for models Sciris covers"/>
</node>
<node CREATED="1501265046215" ID="ID_1224654216" MODIFIED="1501265062912" TEXT="Parameter"/>
<node CREATED="1501265052781" ID="ID_1659675458" MODIFIED="1501265069214" TEXT="ParameterSet"/>
<node CREATED="1501265079230" ID="ID_613992197" MODIFIED="1501265082098" TEXT="ResultSet"/>
<node CREATED="1501265104283" ID="ID_307307909" MODIFIED="1501265122707" TEXT="SimulationResult"/>
<node CREATED="1501105073911" ID="ID_1568535677" MODIFIED="1501105079388" TEXT="SimulationRun">
<node CREATED="1501265297736" MODIFIED="1501265297736" TEXT="a run of a ScirisModel"/>
</node>
<node CREATED="1501105155763" ID="ID_75945432" MODIFIED="1501105159135" TEXT="SimulationBatch">
<node CREATED="1501265306107" MODIFIED="1501265306107" TEXT="a run of a ScirisModel"/>
</node>
</node>
</node>
<node CREATED="1501214425112" ID="ID_749182187" MODIFIED="1501214426815" TEXT="Vue Notes">
<node CREATED="1501178853298" ID="ID_1440293543" MODIFIED="1501178858835" TEXT="vue-cli Commands">
<node CREATED="1501178859616" ID="ID_1545376787" MODIFIED="1501178878049" TEXT="vue list">
<node CREATED="1501178878847" ID="ID_251603371" MODIFIED="1501178938295" TEXT="shows templates vue-cli can build off of"/>
</node>
<node CREATED="1501178885948" ID="ID_732494041" MODIFIED="1501178900906" TEXT="vue init [template] [project name]">
<node CREATED="1501178901717" ID="ID_316857096" MODIFIED="1501178924077" TEXT="makes a project at [project name] based on the [template] template"/>
</node>
</node>
<node CREATED="1501214298855" ID="ID_1720455146" MODIFIED="1501214322480" TEXT="Vue General Info">
<node CREATED="1499713972202" ID="ID_818981018" MODIFIED="1499714301923" TEXT="Vue was released in 2014."/>
<node CREATED="1499726459855" ID="ID_295461384" MODIFIED="1499726471388" TEXT="Vue Version 2 was launched in September, 2016."/>
<node CREATED="1499713965592" ID="ID_1046423053" MODIFIED="1499713969159" TEXT="Angular 1 info">
<node CREATED="1499714304937" ID="ID_1655834723" MODIFIED="1499714314353" TEXT="Angular 1 was released in 2009."/>
</node>
</node>
<node CREATED="1499708853996" ID="ID_361996041" MODIFIED="1501214361855" TEXT="Analysis of desirability of port from Angular 1 to Vue.js">
<font NAME="SansSerif" SIZE="12"/>
<node CREATED="1499707837165" ID="ID_1166115337" MODIFIED="1499712396917" TEXT="Criteria for deciding between options">
<node CREATED="1499707845127" ID="ID_1117332336" MODIFIED="1499707876279" TEXT="We want to develop 3 new front-ends for sites similar to present one."/>
<node CREATED="1499707982567" ID="ID_1025494575" MODIFIED="1499708012815" TEXT="If we be nice if the porting process would take 33% less time if we switch."/>
</node>
<node CREATED="1499707824452" ID="ID_388673024" MODIFIED="1499707828216" TEXT="Main question">
<node CREATED="1499707829046" ID="ID_1714750445" MODIFIED="1499712928061" TEXT="Do we want to switch to Vue.js or stick with Angular 1?"/>
</node>
<node CREATED="1499712309909" ID="ID_1051353019" MODIFIED="1499712323101" TEXT="The two options:">
<node CREATED="1499707876687" FOLDED="true" ID="ID_220101671" MODIFIED="1501214295303" TEXT="Option A (stick with Angular)">
<node CREATED="1499707913392" ID="ID_1513195985" MODIFIED="1499707932471" TEXT="Create 3 new Angular sites from scratch."/>
</node>
<node CREATED="1499707896577" FOLDED="true" ID="ID_1378219361" MODIFIED="1501214295303" TEXT="Option B (port to Vue.js)">
<node CREATED="1499729649190" FOLDED="true" ID="ID_1437314350" MODIFIED="1501214295246" TEXT="Option B1 (&quot;Cold turkey&quot; port)">
<node CREATED="1499729657263" ID="ID_1308527396" MODIFIED="1499729741903" TEXT="Start a Vue site from scratch and migrate over Angular parts until it&apos;s showtime with the Vue.js site."/>
<node CREATED="1499707961411" ID="ID_576923812" MODIFIED="1499731142929" TEXT="Create 3 new Vue sites from scratch based on new site."/>
</node>
<node CREATED="1499729745741" FOLDED="true" ID="ID_386825333" MODIFIED="1501214295247" TEXT="Option B2 (ngVue port)">
<node CREATED="1499729755277" ID="ID_70688160" MODIFIED="1499731202995" TEXT="Start by keeping the website in Angular 1, but use ngVue to incrementally move over Angular parts into Vue."/>
</node>
<node CREATED="1499731085729" FOLDED="true" ID="ID_548970127" MODIFIED="1501214295249" TEXT="Option B3 (mirror &quot;cold turkey&quot; + ngVue port)">
<node CREATED="1499729755277" ID="ID_407106660" MODIFIED="1499731202995" TEXT="Start by keeping the website in Angular 1, but use ngVue to incrementally move over Angular parts into Vue."/>
<node CREATED="1499731215909" ID="ID_1288811339" MODIFIED="1499731238065" TEXT="In parallel, move the new converted Angular parts into the &quot;cold turkey&quot; shell."/>
<node CREATED="1499731241857" ID="ID_894830468" MODIFIED="1499731263334" TEXT="Finally, we will have a working &quot;cold turkey&quot; version, and we can replace the Angular 1 version with it."/>
<node CREATED="1499707961411" ID="ID_1414569894" MODIFIED="1499731142929" TEXT="Create 3 new Vue sites from scratch based on new site."/>
</node>
</node>
</node>
<node CREATED="1499708956412" ID="ID_1418209271" MODIFIED="1499708959473" TEXT="advantages">
<node CREATED="1499709120639" ID="ID_683658226" MODIFIED="1499730475344" TEXT="Vue is less complex and easier to use than Angular 1">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1499712414781" ID="ID_1924885058" MODIFIED="1501214366703" TEXT="Vue has faster performance than Angular 1">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1499712431249" ID="ID_726856677" MODIFIED="1499712455105" TEXT="find justifications and, if possible, benchmarks"/>
<node CREATED="1499726713197" ID="ID_558909228" MODIFIED="1499726736410" TEXT="This is because it doesn&apos;t do dirty checking and go through an inefficient digest cycle."/>
</node>
<node CREATED="1499726835854" ID="ID_1333477404" MODIFIED="1499726842790" TEXT="Vue is even faster than Angular 2."/>
<node CREATED="1499713399214" ID="ID_389119798" MODIFIED="1499713419213" TEXT="Vue is less &quot;opinionated&quot; than Angular, forcing users less to do things its way."/>
<node CREATED="1499726563606" ID="ID_1627744019" MODIFIED="1499726601834" TEXT="It&apos;s newer, so it had the benefits of building the best features off of earlier frameworks into it."/>
<node CREATED="1499729210579" ID="ID_1092435306" MODIFIED="1499730506483" TEXT="Angular 1 is made obsolete by Angular 2, but Angular 2 has been bloated and not well liked, so it&apos;s better to migrate to a non-obsolete framework.">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
</node>
<node CREATED="1499708959911" ID="ID_563834073" MODIFIED="1499708965546" TEXT="disadvantages">
<node CREATED="1499713242958" ID="ID_1902602038" MODIFIED="1499713275807" TEXT="Angular 1 is majorly battle-tested and in wide use, whereas Vue is more untried."/>
<node CREATED="1499729166377" ID="ID_1547259854" MODIFIED="1499729202050" TEXT="Angular 1 currently has a larger pool of developers and community, although Vue is growing."/>
</node>
</node>
</node>
<node CREATED="1501615520375" ID="ID_1564875433" MODIFIED="1501615523975" TEXT="Geppetto framework">
<node CREATED="1501615539301" ID="ID_350912146" LINK="http://www.geppetto.org/" MODIFIED="1501615550103" TEXT="home page"/>
<node CREATED="1501616029682" ID="ID_1545226286" LINK="https://github.com/openworm/org.geppetto" MODIFIED="1501616033796" TEXT="GitHub page"/>
<node CREATED="1501615524869" ID="ID_1490303938" MODIFIED="1501615534509" TEXT="Existing framework for scientific visualization"/>
</node>
</node>
<node CREATED="1501096067528" ID="ID_907174623" MODIFIED="1501096070191" POSITION="left" TEXT="Resources">
<node CREATED="1501214094242" ID="ID_278515612" MODIFIED="1501214096974" TEXT="Project Notebook"/>
<node CREATED="1501214099299" ID="ID_73628816" MODIFIED="1501214105162" TEXT="Files"/>
<node CREATED="1501214116845" ID="ID_1307392466" MODIFIED="1501214119035" TEXT="Web Links">
<node CREATED="1501214144219" ID="ID_924215585" MODIFIED="1501214148436" TEXT="Technical References">
<node CREATED="1501214214187" ID="ID_1973499916" MODIFIED="1501214215067" TEXT="Vue">
<node CREATED="1499738496996" ID="ID_1789409955" MODIFIED="1499738508527" TEXT="Vue.js">
<node CREATED="1496671373820" ID="ID_1611056948" LINK="https://vuejs.org/" MODIFIED="1501220945217" TEXT="Vue home page">
<node CREATED="1499213094617" ID="ID_1903654204" LINK="https://vuejs.org/v2/guide/" MODIFIED="1501220949040" TEXT="Vue Guide documentation">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1499196279067" ID="ID_861922313" LINK="https://vuejs.org/v2/guide/comparison.html" MODIFIED="1499713634274" TEXT="Comparison with other web frameworks (Vue.js page)"/>
</node>
<node CREATED="1499213154038" ID="ID_1777128956" LINK="https://vuejs.org/v2/api/" MODIFIED="1499214039397" TEXT="API documentation">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
</node>
<node CREATED="1499738515796" ID="ID_1048045768" LINK="https://medium.com/vuejs-tips/vue-js-2-2-complete-api-cheat-sheet-c30482a9e936" MODIFIED="1499738524702" TEXT="Vue.js 2.2 complete api cheat sheet"/>
<node CREATED="1499738604074" ID="ID_373180637" LINK="https://stackoverflow.com/questions/41164672/whats-the-equivalent-of-angular-service-in-vuejs" MODIFIED="1499738621552" TEXT="What&apos;s the equivalent of Angular Service in VueJS? (Stack Overflow)"/>
</node>
<node CREATED="1501025494800" ID="ID_654401" MODIFIED="1501025504867" TEXT="routing for Vue">
<node CREATED="1501025506076" ID="ID_1750384254" MODIFIED="1501220951521" TEXT="vue-router library">
<node CREATED="1501025521768" ID="ID_1722078590" LINK="https://router.vuejs.org/en/" MODIFIED="1501025661268" TEXT="vue-router documentation"/>
<node CREATED="1501025577638" ID="ID_53668999" LINK="https://github.com/vuejs/vue-router" MODIFIED="1501025589844" TEXT="GitHub page"/>
</node>
</node>
<node CREATED="1499738340512" ID="ID_1460259132" MODIFIED="1499738348460" TEXT="AJAX for Vue">
<node CREATED="1499738349260" ID="ID_386183068" MODIFIED="1501220925840" TEXT="Axios">
<node CREATED="1499738369312" ID="ID_1186053826" LINK="https://github.com/mzabriskie/axios" MODIFIED="1499738387293" TEXT="GitHub page"/>
<node CREATED="1499738355942" ID="ID_547715790" MODIFIED="1499738363774" TEXT="Evan You recommands this."/>
</node>
</node>
<node CREATED="1499738810498" ID="ID_1195523008" MODIFIED="1499738851498" TEXT="Centralized State Management for Vue">
<node CREATED="1499738816938" ID="ID_1770960297" MODIFIED="1501220927902" TEXT="Vuex">
<node CREATED="1499738819560" ID="ID_1791678174" LINK="https://github.com/vuejs/vuex" MODIFIED="1499738837812" TEXT="GitHub page"/>
<node CREATED="1499739004066" ID="ID_411328992" LINK="http://vuex.vuejs.org/en/intro.html" MODIFIED="1499739032568" TEXT="Vue.js site Vuex page"/>
</node>
</node>
<node CREATED="1499742330324" ID="ID_1623406347" MODIFIED="1499742334670" TEXT="vue-devtools">
<node CREATED="1499742335766" ID="ID_1627589748" LINK="https://github.com/vuejs/vue-devtools" MODIFIED="1499742353005" TEXT="GitHub page"/>
<node CREATED="1499742371066" ID="ID_64650942" MODIFIED="1499742390426" TEXT="There is a Firefox addon, as well as the Chrome tools."/>
</node>
<node CREATED="1501221090317" ID="ID_32521260" MODIFIED="1501221093459" TEXT="vue-loader">
<node CREATED="1501221094227" ID="ID_1792847106" MODIFIED="1501221099424" TEXT="Webpack loader for Vue"/>
<node CREATED="1501221101984" ID="ID_1542447842" LINK="https://vue-loader.vuejs.org/en/" MODIFIED="1501221113635" TEXT="doc page"/>
</node>
<node CREATED="1501294389241" ID="ID_226227028" MODIFIED="1501294391115" TEXT="http-proxy-middleware">
<node CREATED="1501294392658" ID="ID_1904115059" LINK="https://github.com/chimurai/http-proxy-middleware" MODIFIED="1501294408203" TEXT="GitHub page"/>
<node CREATED="1501634809153" ID="ID_1404233600" MODIFIED="1501634831693" TEXT="This needs to be used (through webpack) to make sure the dev server can run with a server."/>
</node>
<node CREATED="1499712787539" ID="ID_1794750378" MODIFIED="1499712794129" TEXT="Advantages of Vue vs. Angular">
<node CREATED="1499716989126" FOLDED="true" ID="ID_363918406" LINK="http://www.evontech.com/what-we-are-saying/entry/why-developers-now-compare-vuejs-to-javascript-giants-angular-and-react.html" MODIFIED="1501220938477" TEXT="Why Do Developers Now Compare Vue.js to JavaScript Giants Angular and React?">
<node CREATED="1499726415028" ID="ID_257296685" MODIFIED="1499726429013" TEXT="article written around 10/25/2016"/>
<node CREATED="1499726835854" ID="ID_1312715534" MODIFIED="1499726863740" TEXT="Vue 2.0 is even faster than Angular 2."/>
<node CREATED="1499726913413" ID="ID_1471641888" LINK="http://stefankrause.net/js-frameworks-benchmark4/webdriver-ts/table.html" MODIFIED="1499727316296" TEXT="has link to benchmarks page (Round 4)"/>
</node>
<node CREATED="1499196279067" FOLDED="true" ID="ID_1012413656" LINK="https://vuejs.org/v2/guide/comparison.html" MODIFIED="1501214295306" TEXT="Comparison with other web frameworks (Vue.js page)">
<node CREATED="1499713718751" ID="ID_949300177" MODIFIED="1499713738261" TEXT="Angular becomes a lot slower when there are many watchers because of the digest cycle."/>
</node>
<node CREATED="1499792498446" FOLDED="true" ID="ID_103185134" LINK="https://superdevelopment.com/2017/04/03/vue-js-the-next-library-for-angular-1-developers/" MODIFIED="1501214295307" TEXT="Vue.js &#x2013; The Next Library for Angular 1 Developers">
<font NAME="SansSerif" SIZE="12"/>
<node CREATED="1499796319980" ID="ID_1068031038" MODIFIED="1499796334433" TEXT="This has a good explanation of the problems with Angular 1."/>
<node CREATED="1499796435233" ID="ID_11732494" MODIFIED="1499796448485" TEXT="An example is given of integrating Vue with an Angular app."/>
</node>
<node CREATED="1499713876992" ID="ID_1933242001" LINK="http://www.valuecoders.com/blog/technology-and-apps/vue-js-comparison-angular-react/" MODIFIED="1499713899914" TEXT="Vue.js Is Good, But Is It Better Than Angular Or React? (Value Coders)"/>
<node CREATED="1499715350021" ID="ID_995598868" LINK="https://www.altexsoft.com/blog/engineering/angularjs-vs-knockout-js-vs-vue-js-vs-backbone-js-which-framework-suits-your-project-best/" MODIFIED="1499715360988" TEXT="AngularJS vs Knockout.js vs Vue.js vs Backbone.js: Which Framework Suits Your Project Best?"/>
<node CREATED="1499713327267" ID="ID_1901261966" LINK="https://stackoverflow.com/questions/37210364/angularjs-vs-vuejs-what-are-the-advantages-and-disadvantages" MODIFIED="1499713350323" TEXT="AngularJs vs VueJS What are the advantages and disadvantages (Stack Overflow)"/>
<node CREATED="1499712806400" ID="ID_1016766723" LINK="https://hashnode.com/post/what-is-the-advantage-of-vuejs-over-angular-or-react-in-terms-of-performance-ease-of-use-and-scalability-cit4sbaxi18xj8k53ejiuahiy" MODIFIED="1499712817944" TEXT="What is the advantage of Vue.js over Angular or React in terms of performance, ease of use and scalability?"/>
<node CREATED="1499712996045" FOLDED="true" ID="ID_163284588" LINK="http://react-etc.net/entry/comparison-js-angular-react-vue" MODIFIED="1501214295310" TEXT="JS Comparison: Angular vs. React vs. Vue">
<node CREATED="1499713091687" ID="ID_131575520" MODIFIED="1499713107942" TEXT="&quot;The largest issue with Angular [1] is that the performance can degrade quite easily.&quot;"/>
<node CREATED="1499713207640" ID="ID_1630755127" MODIFIED="1499713233417" TEXT="This article considers Vue a more uncertain option due to the volatility of the JavaScript world."/>
</node>
<node CREATED="1499727296756" FOLDED="true" ID="ID_1025430441" MODIFIED="1501214295310" TEXT="Performance benchmarks">
<node CREATED="1499726958305" FOLDED="true" ID="ID_285438912" LINK="http://stefankrause.net/js-frameworks-benchmark4/webdriver-ts/table.html" MODIFIED="1501214295254" TEXT="Results for js web frameworks benchmark &#x2013; round 4">
<node CREATED="1499727028895" ID="ID_542067760" LINK="JS web frameworks benchmark &#x2013; Round 4" MODIFIED="1499727036631" TEXT="article this is from"/>
</node>
<node CREATED="1499727321717" ID="ID_714668688" LINK="http://www.stefankrause.net/js-frameworks-benchmark6/webdriver-ts-results/table.html" MODIFIED="1499727334918" TEXT="Results for js web frameworks benchmark &#x2013; round 6"/>
</node>
<node CREATED="1499792534141" ID="ID_293068592" LINK="https://wildermuth.com/2017/02/12/Why-I-Moved-to-Vue-js-from-Angular-2" MODIFIED="1499792561012" TEXT="Why I Moved to Vue.js from Angular 2"/>
</node>
<node CREATED="1499728234449" ID="ID_1033022257" LINK="https://medium.com/dailyjs/how-to-migrate-from-angularjs-to-vue-4a1e9721bea8" MODIFIED="1499730066531" TEXT="How to Migrate from AngularJS to Vue (using ngVue)">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
</node>
<node CREATED="1501214149309" ID="ID_1709939587" MODIFIED="1501214213209" TEXT="npm">
<node CREATED="1501214154088" ID="ID_1892831691" MODIFIED="1501214159035" TEXT="Node Package Manager"/>
<node CREATED="1501214160135" ID="ID_1172457398" LINK="https://docs.npmjs.com/getting-started/what-is-npm" MODIFIED="1501214189791" TEXT="Getting started with npm"/>
</node>
<node CREATED="1501091709140" ID="ID_1971930469" MODIFIED="1501214241221" TEXT="JSFiddle">
<node CREATED="1501091721771" ID="ID_1762125154" LINK="https://jsfiddle.net" MODIFIED="1501091733533" TEXT="link"/>
<node CREATED="1501092004092" ID="ID_293732933" MODIFIED="1501092027373" TEXT="An on-browser tool where I can play with JavaSript, HTML, and CSS"/>
<node CREATED="1501093255294" ID="ID_657394922" LINK="https://jsfiddle.net/epz4h7ez/" MODIFIED="1501093274448" TEXT="first fiddle (Hello world)"/>
</node>
<node CREATED="1501305373798" ID="ID_751825847" MODIFIED="1501305376402" TEXT="JavaScript">
<node CREATED="1501305377336" ID="ID_1900284242" LINK="https://developer.mozilla.org/en-US/docs/Web/JavaScript" MODIFIED="1501376741961" TEXT="MDN JavaScript reference">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
</node>
<node CREATED="1490115655382" ID="ID_1365733481" MODIFIED="1499637971132" TEXT="Flask">
<font NAME="SansSerif" SIZE="12"/>
<node CREATED="1495996911198" ID="ID_1195430302" LINK="http://flask.pocoo.org/" MODIFIED="1499637971132" TEXT="home site">
<node CREATED="1496796650263" ID="ID_736694783" LINK="http://flask.pocoo.org/docs/0.12/" MODIFIED="1496796659268" TEXT="documentation link"/>
<node CREATED="1501704900293" ID="ID_1390409463" LINK="http://flask.pocoo.org/docs/0.12/api/" MODIFIED="1501704935243" TEXT="API docs"/>
</node>
<node CREATED="1490125502142" ID="ID_734985344" MODIFIED="1490125521886" TEXT="Python web application micro-framework"/>
<node CREATED="1495989224004" ID="ID_1283183120" MODIFIED="1495989243783" TEXT="based on WSGI"/>
<node CREATED="1496950842898" ID="ID_802053029" MODIFIED="1499637971132" TEXT="ancillary packages">
<node CREATED="1496950848964" ID="ID_987745873" MODIFIED="1499637971132" TEXT="Flask-Login">
<node CREATED="1497198071295" ID="ID_1315828578" MODIFIED="1497198080746" TEXT="manages user log-in"/>
<node CREATED="1496950863569" ID="ID_438714619" LINK="https://flask-login.readthedocs.io/en/latest/" MODIFIED="1497198063820" TEXT="documentation, including API"/>
</node>
<node CREATED="1496960926331" ID="ID_171871692" MODIFIED="1499637971132" TEXT="Flask-SQLAlchemy">
<node CREATED="1496960913186" ID="ID_1098493741" LINK="http://flask-sqlalchemy.pocoo.org/2.1/" MODIFIED="1497198059170" TEXT="documentation"/>
</node>
<node CREATED="1497199019078" ID="ID_1010945001" MODIFIED="1499637971132" TEXT="Flask-RESTful">
<node CREATED="1497199024268" ID="ID_357084385" LINK="https://flask-restful.readthedocs.io/en/0.3.5/" MODIFIED="1497199038535" TEXT="documentation"/>
</node>
</node>
</node>
<node CREATED="1495989174185" ID="ID_657134353" MODIFIED="1499637971132" TEXT="Python Web Server Gateway Interface (WSGI)">
<node CREATED="1495989187834" ID="ID_1050864948" LINK="http://legacy.python.org/dev/peps/pep-0333/" MODIFIED="1495989204571" TEXT="specification site"/>
</node>
<node CREATED="1490115416151" ID="ID_949420009" MODIFIED="1499637971132" TEXT="Twisted">
<node CREATED="1490126117036" ID="ID_1834030967" LINK="https://en.wikipedia.org/wiki/Twisted_(software)" MODIFIED="1490126132909" TEXT="Twisted (software) (Wikipedia article)"/>
<node CREATED="1495669615546" ID="ID_1222680627" MODIFIED="1499637971132" TEXT="Twisted home page">
<node CREATED="1495209768465" ID="ID_828265196" LINK="https://twistedmatrix.com/trac/" MODIFIED="1495669632425" TEXT="main site link"/>
<node CREATED="1495669751540" ID="ID_1849595302" LINK="http://twistedmatrix.com/documents/current/core/howto/index.html" MODIFIED="1499637971132" TEXT="Developer Guides link">
<node CREATED="1495670694646" ID="ID_33671910" MODIFIED="1495670705667" TEXT="This is probably the best place to start for a tutorial."/>
</node>
<node CREATED="1495670418147" ID="ID_1664317651" LINK="http://twistedmatrix.com/documents/current/core/howto/endpoints.html" MODIFIED="1495670432735" TEXT="HOWTO Page on endpoints"/>
<node CREATED="1495669653809" ID="ID_754360218" LINK="http://twistedmatrix.com/documents/current/api/" MODIFIED="1495669660637" TEXT="API Documentation"/>
<node CREATED="1495988934582" ID="ID_1252768715" LINK="http://crossbario.com/blog/Going-Asynchronous-from-Flask-to-Twisted-Klein/" MODIFIED="1499637971132" TEXT="Going asynchronous: from Flask to Twisted Klein">
<node CREATED="1495988953984" ID="ID_744576034" MODIFIED="1495988970418" TEXT="This basically explains what the rationale is of using Twisted and Flask together."/>
</node>
</node>
<node CREATED="1495160134074" ID="ID_899691452" MODIFIED="1495160143182" TEXT="This acts like an Apache server."/>
</node>
<node CREATED="1501776727211" ID="ID_331782536" MODIFIED="1501776731523" TEXT="mpld3">
<node CREATED="1490113813130" ID="ID_928809867" MODIFIED="1499637971102" TEXT="Matplotlib D3">
<node CREATED="1490113819834" ID="ID_1029673621" LINK="http://mpld3.github.io/" MODIFIED="1502049397621" TEXT="project home page">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1502048129783" ID="ID_552468356" LINK="https://github.com/mpld3/mpld3" MODIFIED="1502048134853" TEXT="GitHub source"/>
<node CREATED="1501786621870" ID="ID_860725635" LINK="http://mpld3.github.io/modules/API.html" MODIFIED="1501788665287" TEXT="API documentation"/>
<node CREATED="1490113868626" MODIFIED="1490113868626" TEXT="The mpld3 project brings together Matplotlib, the popular Python-based graphing library, and D3js, the popular JavaScript library for creating interactive data visualizations for the web."/>
</node>
<node CREATED="1495215936719" ID="ID_1345559648" MODIFIED="1499637971102" TEXT="D3.js">
<node CREATED="1495215942293" ID="ID_245316578" LINK="https://en.wikipedia.org/wiki/D3.js" MODIFIED="1495215948880" TEXT="Wikipedia article"/>
</node>
</node>
<node CREATED="1501791330919" ID="ID_1847189861" MODIFIED="1501791334222" TEXT="matplotlib">
<node CREATED="1291670225502" ID="ID_1069109518" LINK="http://matplotlib.sourceforge.net/" MODIFIED="1313532216725" TEXT="matplotlib / pylab documentation">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
</node>
<node CREATED="1501796731267" ID="ID_427065587" MODIFIED="1501796732992" TEXT="Webpack">
<node CREATED="1501797397851" ID="ID_735865464" LINK="https://webpack.js.org/" MODIFIED="1501797405347" TEXT="home page"/>
</node>
<node CREATED="1281730745976" ID="ID_1314358534" MODIFIED="1492448223642" TEXT="Python">
<node CREATED="1390062725281" ID="ID_392025075" LINK="http://pandas.pydata.org/pandas-docs/stable/" MODIFIED="1469898148477" TEXT="pandas documentation">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1390062873706" ID="ID_309739533" MODIFIED="1390062877573" TEXT="data analysis tools"/>
<node CREATED="1399850007412" ID="ID_901153746" MODIFIED="1399850022565" TEXT="Note: pandas only works under CPython"/>
</node>
<node CREATED="1296424235211" ID="ID_1912405715" LINK="http://docs.scipy.org/doc/" MODIFIED="1469898148493" TEXT="numpy / scipy documentation">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
<node CREATED="1390062891389" ID="ID_1195237504" MODIFIED="1390062919515" TEXT="scientific programming using efficient homogeneous arrays"/>
<node CREATED="1399850269229" ID="ID_1674825132" MODIFIED="1399850286150" TEXT="Note: numpy and scipy have supposedly been ported to IronPython"/>
</node>
<node CREATED="1386898400602" ID="ID_1391199782" LINK="http://scikit-learn.org/stable/documentation.html" MODIFIED="1386898426594" TEXT="scikit-learn documentation">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1407895486824" ID="ID_1241269126" LINK="http://statsmodels.sourceforge.net/stable/" MODIFIED="1407895541355" TEXT="statsmodels documentation">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1276273923735" ID="ID_436642041" LINK="http://www.korokithakis.net/tutorials/python" MODIFIED="1469898148495" TEXT="Learn Python in 10 minutes">
<font NAME="SansSerif" SIZE="12"/>
<node CREATED="1276273935581" ID="ID_405816665" MODIFIED="1277569116174" TEXT="sent by Bill"/>
</node>
<node CREATED="1271645043406" ID="ID_1921969110" LINK="http://www.python.org/" MODIFIED="1277569116174" TEXT="Python Programming Language -- Official Website"/>
<node CREATED="1358541820647" ID="ID_1278243714" LINK="http://docs.python.org/2/index.html" MODIFIED="1492448240111" TEXT="Python 2.7.13 documentation">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1275943690062" ID="ID_1064375908" LINK="http://docs.python.org/tutorial/" MODIFIED="1452102636553" TEXT="Python Tutorial (version 2.7.11)"/>
<node CREATED="1281058809275" ID="ID_553740466" LINK="http://vefur.simula.no/intro-programming/" MODIFIED="1281058908375" TEXT="Primer on Scientific Programming book site"/>
<node CREATED="1291309485130" ID="ID_1360930487" LINK="http://mathesaurus.sourceforge.net/matlab-numpy.html" MODIFIED="1291309507630" TEXT="NumPy for MATLAB users"/>
<node CREATED="1369161450833" ID="ID_1500198662" LINK="http://www.stereoplex.com/blog/understanding-imports-and-pythonpath" MODIFIED="1369161476851" TEXT="Understanding imports and PYTHONPATH"/>
<node CREATED="1294326590975" ID="ID_1245914571" LINK="http://matplotlib.sourceforge.net/users/event_handling.html" MODIFIED="1294326620433" TEXT="matplotlib event handling and picking"/>
<node CREATED="1296269248224" ID="ID_527551733" LINK="http://matplotlib.sourceforge.net/api/index.html" MODIFIED="1296269277741" TEXT="matplotlib API"/>
<node CREATED="1296269215348" ID="ID_73902524" LINK="http://dsnra.jpl.nasa.gov/software/Python/python-modules/matplotlib.image.html" MODIFIED="1469898148495" TEXT="matplotlib.image documentation">
<node CREATED="1296269232403" ID="ID_1380059089" MODIFIED="1296269245028" TEXT="for some reason this is not in the main documentation from the link above"/>
</node>
<node CREATED="1294325795370" ID="ID_1901888266" LINK="http://www.scipy.org/Cookbook/Matplotlib#head-18cc54c47ffe266c322ee2c6b1c203f515f77d4d" MODIFIED="1294325817168" TEXT="Cookbook / matplotlib"/>
<node CREATED="1297127256959" ID="ID_1873362109" LINK="http://www.scipy.org/Cookbook/Matplotlib/Animations" MODIFIED="1297127277292" TEXT="Cookbook / pylab animations"/>
<node CREATED="1313519993647" ID="ID_1876036284" LINK="http://docs.scipy.org/doc/scipy/reference/signal.html" MODIFIED="1313520004957" TEXT="scipy.signal documentation"/>
<node CREATED="1323114179276" ID="ID_666397082" LINK="http://mlabwrap.sourceforge.net/" MODIFIED="1323114212713" TEXT="mlabwrap documentation"/>
<node CREATED="1357939180627" ID="ID_1196134545" LINK="http://ilab.cs.byu.edu/python/" MODIFIED="1357939199899" TEXT="Python Network Programming">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1383686911900" ID="ID_971797808" LINK="http://www.tutorialspoint.com/python/python_multithreading.htm" MODIFIED="1383686923703" TEXT="Python Multithreaded Programming">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1358194346854" ID="ID_1824111499" MODIFIED="1469898148496" TEXT="VPython visualization package">
<node CREATED="1358181330763" ID="ID_1376972768" LINK="http://www.vpython.org/" MODIFIED="1358194364503" TEXT="main site">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
<node CREATED="1358194365451" ID="ID_216801327" LINK="http://www.vpython.org/webdoc/visual/index.html" MODIFIED="1358194386787" TEXT="documentation">
<font BOLD="true" NAME="SansSerif" SIZE="12"/>
</node>
</node>
<node CREATED="1366299377689" ID="ID_1874676952" LINK="http://networkx.lanl.gov/pygraphviz/" MODIFIED="1366299530063" TEXT="graphviz Python interface (PyGraphviz)"/>
<node CREATED="1460166042393" ID="ID_1856493656" MODIFIED="1469898148497" TEXT="IPython">
<node CREATED="1460166340582" ID="ID_1867081524" LINK="http://ipython.org/notebook.html" MODIFIED="1460166355528" TEXT="The Jupyter Notebook"/>
<node CREATED="1460166060106" ID="ID_1681679283" MODIFIED="1469898148497" TEXT="Jupyter Notebooks (formerly called IPython Notebooks)">
<node CREATED="1460166065116" ID="ID_1871173911" LINK="http://nbviewer.jupyter.org/" MODIFIED="1460166164027" TEXT="link"/>
</node>
</node>
<node CREATED="1470878910868" ID="ID_721661921" MODIFIED="1470878913914" TEXT="Python vs. R">
<node CREATED="1470878943074" ID="ID_1734893534" LINK="https://www.reddit.com/r/Python/comments/2tkkxd/considering_putting_my_efforts_into_python/" MODIFIED="1470878956179" TEXT="Considering putting my efforts into Python instead of R, need advice">
<node CREATED="1470878958060" ID="ID_1636924831" MODIFIED="1470878963311" TEXT="first reply to this is really good"/>
</node>
</node>
<node CREATED="1472166168715" ID="ID_199570850" MODIFIED="1472166173907" TEXT="Web application framework">
<node CREATED="1472166176279" ID="ID_741662530" LINK="http://multithreaded.stitchfix.com/blog/2015/07/16/pyxley/" MODIFIED="1472166200857" TEXT="Pyxley: Python Powered Dashboards">
<node CREATED="1472169921272" ID="ID_1302478929" MODIFIED="1472169937030" TEXT="Shiny-like apps"/>
</node>
<node CREATED="1472166769977" ID="ID_548358070" MODIFIED="1472166772528" TEXT="flask">
<node CREATED="1472169587394" ID="ID_1077242250" LINK="http://flask.pocoo.org/" MODIFIED="1472169611929" TEXT="Flask home page"/>
<node CREATED="1472166774537" ID="ID_327761116" LINK="https://en.wikipedia.org/wiki/Flask_(web_framework)" MODIFIED="1472166789912" TEXT="Wikipedia link"/>
</node>
</node>
</node>
</node>
<node CREATED="1501263086130" ID="ID_46396829" LINK="https://gchadder3.shinyapps.io/polyfitter/" MODIFIED="1501263108914" TEXT="polynomial fitting Shiny app example"/>
</node>
</node>
</node>
</map>
