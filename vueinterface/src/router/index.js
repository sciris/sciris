// index.js -- vue-router path configuration code
//
// Last update: 9/10/17 (gchadder3)

import Vue from 'vue'
import Router from 'vue-router'
import MyPage from '@/components/MyPage'
import LoginPage from '@/components/LoginPage'
import Hello from '@/components/Hello'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'MyPage',
      component: MyPage
    },
    {
      path: '/login',
      name: 'LoginPage',
      component: LoginPage
    },
    {
      path: '/vueinfo',
      name: 'Hello',
      component: Hello
    }
  ]
})
