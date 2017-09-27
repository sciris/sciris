// index.js -- vue-router path configuration code
//
// Last update: 9/27/17 (gchadder3)

import Vue from 'vue'
import Router from 'vue-router'
import MyPage from '@/components/MyPage'
import LoginPage from '@/components/LoginPage'
import MainAdminPage from '@/components/MainAdminPage'
import RegisterPage from '@/components/RegisterPage'
import UserChangeInfoPage from '@/components/UserChangeInfoPage'
import ChangePasswordPage from '@/components/ChangePasswordPage'
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
      path: '/mainadmin',
      name: 'MainAdminPage',
      component: MainAdminPage
    },
    {
      path: '/register',
      name: 'RegisterPage',
      component: RegisterPage
    },
    {
      path: '/changeinfo',
      name: 'UserChangeInfoPage',
      component: UserChangeInfoPage
    },
    {
      path: '/changepassword',
      name: 'ChangePasswordPage',
      component: ChangePasswordPage
    },
    {
      path: '/vueinfo',
      name: 'Hello',
      component: Hello
    },
    { 
      path: '*', 
      redirect: '/' 
    }
  ]
})
