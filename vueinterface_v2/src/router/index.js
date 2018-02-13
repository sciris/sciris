// index.js -- vue-router path configuration code
//
// Last update: 2/12/18 (gchadder3)

import Vue from 'vue'
import Router from 'vue-router'
import ProjectsPage from '@/components/ProjectsPage'
import MyPage from '@/components/MyPage'
import LoginPage from '@/components/LoginPage'
import MainAdminPage from '@/components/MainAdminPage'
import RegisterPage from '@/components/RegisterPage'
import UserChangeInfoPage from '@/components/UserChangeInfoPage'
import ChangePasswordPage from '@/components/ChangePasswordPage'

Vue.use(Router)

export default new Router({
  routes: [
    {
      path: '/',
      name: 'ProjectsPage',
      component: ProjectsPage
    },
    {
      path: '/mypage',
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
      path: '*', 
      redirect: '/' 
    }
  ]
})
