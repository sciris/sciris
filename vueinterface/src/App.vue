<!-- 
App.vue -- App component, the main page

Last update: 9/27/17 (gchadder3)
-->

<template>
  <div id="app">
    <!-- Title bar -->
    <h1>Scatterplotter for Vue</h1>

    <!-- router-link menu -->
    <label>Pages:</label>
    <span v-if="userloggedin()">
      <router-link to="/" exact>
        Main Page
      </router-link> 
      &nbsp;
    </span>
    <span v-if="adminloggedin()">
      <router-link to="/mainadmin" exact>
        Admin Page
      </router-link> 
      &nbsp;
    </span>
    <span v-if="!userloggedin()">
      <router-link to="/login">
        Login Page
      </router-link> 
      &nbsp;
    </span>
    <span v-if="!userloggedin()">
      <router-link to="/register">
        Registration Page
      </router-link> 
      &nbsp;
    </span>
    <router-link to="/vueinfo">
      Vue Info
    </router-link> 
    &nbsp;
    <span v-if="userloggedin()">
      <router-link to="/changeinfo" exact>
        Change Account Info
      </router-link> 
      &nbsp;
      <router-link to="/changepassword" exact>
        Change Password
      </router-link> 
      &nbsp;
    </span>

    <!-- Display of logged in user -->
    <label>Logged In User:</label>
    <span v-if="userloggedin()">
      {{ currentuser.displayname }}
    </span>
    <span v-else>
      None
    </span>    
    &nbsp;

    <!-- Logout button -->
    <button v-if="userloggedin()" @click="logout">Log Out</button>

<!--    <button @click="getallusersinfo">User List</button>
    <button @click="admingetuserinfo">Admin Get User</button>
    <button @click="admindeleteuser">Admin Delete User</button> -->

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
    currentuser () {
      return this.$store.state.currentuser
    }
  },

  created () {
    this.getUserInfo()
  },

  methods: {
    userloggedin () {
      if (this.currentuser.displayname == undefined) 
        return false
      else
        return true
    }, 

    adminloggedin () {
      if (this.userloggedin) {
        return this.currentuser.admin
      }
    },

/*    getallusersinfo () {
      rpcservice.rpcAllGetUsersInfo('get_all_users')
    },

    admingetuserinfo () {
      rpcservice.rpcAdminGetUserInfo('admin_get_user_info', 'admin')
    },

    admindeleteuser () {
      rpcservice.rpcAdminUserCall('admin_delete_user', 'ed')
    }, */

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
      rpcservice.rpcGetCurrentUserInfo('get_current_user_info')
      .then(response => {
        // Set the username to what the server indicates.
        this.$store.commit('newuser', response.data.user)
      })
      .catch(error => {
        // Set the username to {}.  An error probably means the 
        // user is not logged in.
        this.$store.commit('newuser', {})
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
