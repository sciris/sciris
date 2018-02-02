<!-- 
App.vue -- App component, the main page

Last update: 2/2/18 (gchadder3)
-->

<template>
  <div id="app">
    <!-- The top bar, including app title and menu -->
    <top-bar></top-bar>

    <!-- Switchable view -->
    <router-view></router-view>
  </div>
</template>

<script>
import TopBar from '@/components/TopBar'

export default {
  name: 'app', 

  components: {
    TopBar
  }
}
</script>

<!-- Global SCSS/SASS settings go here. -->
<style lang="scss">
  @import './sass/main.scss';

/* #app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
} */
</style>
