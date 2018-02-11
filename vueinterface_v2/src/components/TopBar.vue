<!-- 
TopBar.vue -- TopBar Vue component

Last update: 2/11/18 (gchadder3)
-->

<template>
  <div class="TopBar">
    <div class="elastic header">
      <div style="display:table-cell; width:160px">
        <img src="../assets/images/optima-logo-sciris.png" height="50">
      </div>

      <!-- Logout button (admittedly looks ugly here) -->
      <button v-if="userloggedin()" @click="logout">Log Out</button>

    </div>
  </div>
</template>

<script>
import rpcservice from '../services/rpc-service'
import router from '../router'

export default {
  name: 'TopBar', 

  computed: {
    currentuser() {
      return this.$store.state.currentuser
    }
  },

  created() {
    this.getUserInfo()
  },

  methods: {
    userloggedin() {
      if (this.currentuser.displayname == undefined) 
        return false
      else
        return true
    }, 

    adminloggedin() {
      if (this.userloggedin) {
        return this.currentuser.admin
      }
    },

    logout() {
      // Do the logout request.
      rpcservice.rpcLogoutCall('user_logout')
      .then(response => {
        // Update the user info.
        this.getUserInfo()

        // Navigate to the login page automatically.
        router.push('/login')
      })
    },

    getUserInfo() {
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

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
