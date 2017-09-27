<!-- 
MainAdminPage.vue -- Vue component for the main admin page

Last update: 9/27/17 (gchadder3)
-->

<template>
  <div id="app">
    <p>Scatterplotter Users</p>
    {{ usersList }}
  </div>
</template>

<script>
import rpcservice from '../services/rpc-service'
import router from '../router'

export default {
  name: 'MainAdminPage', 

  data () {
    return {
      usersList: []
    }
  },

  created () {
    this.getUsersInfo()
  },

  methods: {
    getUsersInfo () {
      rpcservice.rpcAllGetUsersInfo('get_all_users')
      .then(response => {
        this.usersList = response.data
      })
      .catch(error => {
      })
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style scoped>
</style>
