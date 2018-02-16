<!-- 
ProjectsPage.vue -- ProjectsPage Vue component

Last update: 2/15/18 (gchadder3)
-->

<template>
  <div class="SitePage">
    <!-- Doodle pane for exploring how SASS styling works. -->
    <div v-show="false" class="DoodleArea">
      <p>a paragraph</p>
    </div>

    <div class="PageSection">
      <h2>Create projects</h2>

      <div class="ControlsRowLabel">
        Choose a demonstration project from our database:
      </div>

      <div class="ControlsRow">
        <select v-model="selectedDemoProject">
          <option v-for="choice in demoProjectList">
            {{ choice }}
          </option>
        </select>
        &nbsp; &nbsp;
        <button class="btn" @click="addDemoProject">Add this project</button>
      </div>

      <div class="ControlsRowLabel">
        Or create/upload a new project:
      </div>

      <div class="ControlsRow">
        <button class="btn" @click="createNewProject">Create new project</button>
        &nbsp; &nbsp;
        <button class="btn" @click="uploadProjectFromFile">Upload project from file</button>
        &nbsp; &nbsp;
        <button class="btn" @click="uploadProjectFromSpreadsheet">Upload project from spreadsheet</button>
      </div>
    </div>

    <div class="PageSection"
         v-if="projectSummaries.length > 0">
      <h2>Manage projects</h2>

      <input type="text" 
             class="txbox" 
             style="margin-bottom: 20px" 
             :placeholder="filterPlaceholder"
             v-model="filterText"/>

      <table class="table table-bordered table-hover table-striped" style="width: auto">
        <thead>
          <tr>
            <th>
              <input type="checkbox"/>
            </th>
            <th>
              Name
              <i class="fas fa-caret-down"></i>
            </th>
            <th>Select</th>
            <th>Created on</th>
            <th>Updated on</th>
            <th>Data uploaded on</th>
            <th>Actions</th>
            <th>Data spreadsheet</th>
            <th>Project file</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="projectSummary in projectSummaries">
            <td>
              <input type="checkbox"/>
            </td>
            <td>{{ projectSummary.projectName }}</td>
            <td>
              <button class="btn __green" @click="openProject(projectSummary.projectName)">Open</button>
            </td>
            <td>{{ projectSummary.creationTime }}</td>
            <td>{{ projectSummary.updateTime ? projectSummary.updateTime: 
              'No modification' }}</td>
            <td>{{ projectSummary.spreadsheetUploadTime ?  projectSummary.spreadsheetUploadTime: 
              'No data uploaded' }}</td>
            <td style="white-space: nowrap">
              <button class="btn" @click="copyProject">Copy</button>
              <button class="btn" @click="renameProject">Rename</button>
            </td>
            <td style="white-space: nowrap">
              <button class="btn" @click="uploadSpreadsheetToProject">Upload</button>
              <button class="btn" @click="downloadSpreadsheetFromProject">Download</button>
            </td>
            <td style="white-space: nowrap">
              <button class="btn" @click="downloadProjectFile">Download</button>
              <button class="btn" @click="downloadProjectFileWithResults">Download with results</button>
            </td>
          </tr>
        </tbody>
      </table>

      <div class="ControlsRow">
        <button class="btn" @click="deleteSelectedProjects">Delete selected</button>
        &nbsp; &nbsp;
        <button class="btn" @click="downloadSelectedProjects">Download selected</button>
      </div>
    </div>
  </div>
</template>

<script>
import axios from 'axios'
var filesaver = require('file-saver')
import rpcservice from '../services/rpc-service'
import router from '../router'

export default {
  name: 'ProjectsPage',

  data() {
    return {
      demoProjectList: [],
      selectedDemoProject: '',
      filterText: '',
      projectSummaries: [],
      demoProjectSummaries: 
        [
          {
            projectName: 'Graph 1', 
            creationTime: '2017-Sep-21 08:44 AM',
            updateTime: '2017-Sep-21 08:44 AM',
            spreadsheetUploadTime: '2017-Sep-21 08:44 AM'
          }, 
          {
            projectName: 'Graph 2',
            creationTime: '2017-Sep-22 08:44 AM',
            updateTime: '',
            spreadsheetUploadTime: '2017-Sep-22 08:44 AM'
          }, 
          {
            projectName: 'Graph 3',
            creationTime: '2017-Sep-23 08:44 AM',
            updateTime: '2017-Sep-23 08:44 AM',
            spreadsheetUploadTime: ''
          }
        ],
      filterPlaceholder: '\u{1f50e} Filter Projects',
      selectedgraph: '',
      serverresponse: '',
      loadedfile: '', 
      servererror: '',
      resourcechoices: []
    }
  },

  created() {
    // If we have no user logged in, automatically redirect to the login page.
    if (this.$store.state.currentuser.displayname == undefined) {
      router.push('/login')
    } 

    // Otherwise...
    else {
      // Initialize the demoProjectList by picking out the project names.
      this.demoProjectList = this.demoProjectSummaries.map(demoProj => demoProj.projectName)

      // Initialize the selection of the demo project to the first element.
      this.selectedDemoProject = this.demoProjectList[0]
    }
  },

  methods: {
    addDemoProject() {
      console.log('addDemoProject() called')

      // way to do a deep copy...
      // let newObj = JSON.parse(JSON.stringify(obj));

      // Should I be doing some kind of copy here or is that implicit?
      var found = this.demoProjectSummaries.find(demoProj => 
        demoProj.projectName == this.selectedDemoProject)
      this.projectSummaries.push(found)

//      this.projectSummaries.push(this.demoProjectSummaries[0])
    },

    createNewProject() {
      console.log('createNewProject() called')
    },

    uploadProjectFromFile() {
      console.log('uploadProjectFromFile() called')
    },

    uploadProjectFromSpreadsheet() {
      console.log('uploadProjectFromSpreadsheet() called')
    },

    openProject(projectName) {
      console.log('openProject() called for ' + projectName)
    },

    copyProject() {
      console.log('copyProject() called')
    },

    renameProject() {
      console.log('renameProject() called')
    },

    uploadSpreadsheetToProject() {
      console.log('uploadSpreadsheetToProject() called')
    },

    downloadSpreadsheetFromProject() {
      console.log('downloadSpreadsheetFromProject() called')
    },

    downloadProjectFile() {
      console.log('downloadProjectFile() called')
    },

    downloadProjectFileWithResults() {
      console.log('downloadProjectFileWithResults() called')
    },

    deleteSelectedProjects() {
      console.log('deleteSelectedProjects() called')
    },

    downloadSelectedProjects() {
      console.log('downloadSelectedProjects() called')
    }
  }
}
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->
<style lang="scss" scoped>
  $vue-blue: #32485F;
  $vue-green: #00C185;

  .DoodleArea {
    height: 200px;
    width: 260px;
    border: 1px solid black;
  }

  .DoodleArea p {
    width: 200px;
    padding: 10px;
    margin: 20px;
//    box-sizing: content-box;
  }

/*  .PageSection h2 {
    color: red;
  } */
/*  h2 {
    color: $vue-blue;
  } */
</style>
