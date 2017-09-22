<!-- 
App.vue -- App component, the main page

Last update: 9/21/17 (gchadder3)
-->

<template>
  <div id="app">
    <!-- Title bar -->
    <h1>Scatterplotter for Vue</h1>

    <!-- router-link menu -->
    <label>Pages:</label>
    <span v-if="username != 'None'">
      <router-link to="/" exact>
        Main Page
      </router-link> 
      &nbsp;
    </span>
    <span v-if="username == 'None'">
      <router-link to="/login">
        Login Page
      </router-link> 
      &nbsp;
    </span>
    <span v-if="username == 'None'">
      <router-link to="/register">
        Registration Page
      </router-link> 
      &nbsp;
    </span>
    <router-link to="/vueinfo">
      Vue Info
    </router-link> 
    &nbsp;

    <!-- Display of logged in user -->
    <label>Logged In User:</label>
    {{ username }}
    &nbsp;

    <!-- Logout button -->
    <button v-if="username != 'None'" @click="logout">Log Out</button>

    <hr/>

    <!-- Switchable view -->
    <router-view></router-view>
  </div>
</template>

<script>
import rpcservice from './services/rpc-service'
import router from './router'

export default {
  name: 'app', 

  computed: {
    username () {
      return this.$store.state.username
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

        // Navigate to the login page automatically.
        router.push('/login')
      })
    },

    getUserInfo () {
      rpcservice.rpcGetUserInfo('get_current_user_info')
      .then(response => {
        // Set the username to what the server indicates.
        this.$store.commit('newuser', response.data.username)
      })
      .catch(error => {
        // Set the username to 'None'.  An error probably means the 
        // user is not logged in.
        this.$store.commit('newuser', 'None')
      })
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
