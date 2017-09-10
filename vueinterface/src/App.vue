<!-- 
App.vue -- App component, the main page

Last update: 9/10/17 (gchadder3)
-->

<template>
  <div id="app">
    <!-- Title bar -->
    <h1>Scatterplotter for Vue</h1>

    <!-- router-link menu -->
    <label>Pages:</label>
    <router-link to="/">
      Main Page
    </router-link> &nbsp;
    <router-link to="/login">
      Login Page
    </router-link> &nbsp;
    <router-link to="/vueinfo">
      Vue Info
    </router-link> &nbsp;

    <label>Logged In User:</label>
    {{ username }}

    &nbsp;

    <button @click="logout">Log Out</button>

    <hr/>

    <!-- Switchable view -->
    <router-view></router-view>
  </div>
</template>

<script>
import rpcservice from './services/rpc-service'

export default {
  name: 'app', 

  data () {
    return {
      username: 'None'
    }
  },

  created () {
    this.getUserInfo()
  },

  methods: {
    logout () {
      // Do the logout request.
      rpcservice.rpcLogoutCall('user_logout')
      .then(response => {
        // Update the user info.
        this.getUserInfo()
      })
    },

    getUserInfo () {
      rpcservice.rpcGetUserInfo('get_current_user_info')
      .then(response => {
        // Set the username to what the server indicates.
        this.username = response.data.username
      })
      .catch(error => {
        // Set the username to 'None'.  An error probably means the 
        // user is not logged in.
        this.username = 'None'
      })
    },

    goMainPage () {
      console.log('go to the main page')
    }, 

    goVueInfoPage () {
      console.log('go to the Vue info page')
    }
  }
}
</script>

<style>
/* #app {
  font-family: 'Avenir', Helvetica, Arial, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
  text-align: center;
  color: #2c3e50;
  margin-top: 60px;
} */
</style>
