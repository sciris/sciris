<!-- 
UserChangeInfoPage.vue -- Vue component for a page to change account info

Last update: 1/29/18 (gchadder3)
-->

<template>
  <div class="SitePage">
    <label>Username:</label>
    <input v-model='changeUserName'/>
    <br/>

    <label>Display Name:</label>
    <input v-model='changeDisplayName'/>
    <br/>

    <label>Email:</label>
    <input v-model='changeEmail'/>
    <br/>

    <label>Reenter Password (to validate):</label>
    <input v-model='changePassword'/>
    <br/>

    <button @click="tryChangeInfo">Update</button>
    <br/>

    <p v-if="changeResult != ''">{{ changeResult }}</p>
  </div>
</template>

<script>
import rpcservice from '../services/rpc-service'
import router from '../router'

export default {
  name: 'UserChangeInfoPage', 

  data () {
    return {
      changeUserName: this.$store.state.currentuser.username,
      changeDisplayName: this.$store.state.currentuser.displayname,
      changeEmail: this.$store.state.currentuser.email,
      changePassword: '',
      changeResult: ''
    }
  }, 

  methods: {
    tryChangeInfo () {
      rpcservice.rpcUserChangeInfoCall('user_change_info', this.changeUserName, 
        this.changePassword, this.changeDisplayName, this.changeEmail)
      .then(response => {
        if (response.data == 'success') {
          // Set a success result to show.
          this.changeResult = 'Success!'

          // Read in the full current user information.
          rpcservice.rpcGetCurrentUserInfo('get_current_user_info')
          .then(response2 => {
            // Set the username to what the server indicates.
            this.$store.commit('newuser', response2.data.user)

            // Navigate automatically to the home page.
            router.push('/')
          })
          .catch(error => {
            // Set the username to {}.  An error probably means the 
            // user is not logged in.
            this.$store.commit('newuser', {})
          })
        } else {
          // Set a failure result to show.
          this.changeResult = 'Change of account info failed.'
        }
      })
      .catch(error => {
        this.changeResult = 'Server error.  Please try again later.'
      })
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
