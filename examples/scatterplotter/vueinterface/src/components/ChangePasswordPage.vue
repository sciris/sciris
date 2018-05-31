<!-- 
ChangePasswordPage.vue -- Vue component for a page to change password

Last update: 1/29/18 (gchadder3)
-->

<template>
  <div class="SitePage">
    <label>New Password:</label>
    <input v-model='newPassword'/>
    <br/>

    <label>Reenter Old Password (to validate):</label>
    <input v-model='oldPassword'/>
    <br/>

    <button @click="tryChangePassword">Update</button>
    <br/>

    <p v-if="changeResult != ''">{{ changeResult }}</p>
  </div>
</template>

<script>
import rpcservice from '../services/rpc-service'
import router from '../router'

export default {
  name: 'ChangePasswordPage', 

  data () {
    return {
      oldPassword: '',
      newPassword: '',
      changeResult: ''
    }
  }, 

  methods: {
    tryChangePassword () {
      rpcservice.rpcChangePasswordCall('user_change_password', this.oldPassword, 
        this.newPassword)
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
          this.changeResult = 'Change of password failed.'
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
