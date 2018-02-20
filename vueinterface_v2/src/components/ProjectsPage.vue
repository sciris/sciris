<!-- 
ProjectsPage.vue -- ProjectsPage Vue component

Last update: 2/19/18 (gchadder3)
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
              <input type="checkbox" @click="selectAll()" v-model="allSelected"/>
            </th>
            <th @click="updateSorting('name')" class="sortable">
              Name
              <span v-show="sortColumn == 'name' && !sortReverse">
                <i class="fas fa-caret-down"></i>
              </span>
              <span v-show="sortColumn == 'name' && sortReverse">
                <i class="fas fa-caret-up"></i>
              </span>
              <span v-show="sortColumn != 'name'">
                <i class="fas fa-caret-up" style="visibility: hidden"></i>
              </span>
            </th>
            <th>Select</th>
            <th @click="updateSorting('creationTime')" class="sortable">
              Created on
              <span v-show="sortColumn == 'creationTime' && !sortReverse">
                <i class="fas fa-caret-down"></i>
              </span>
              <span v-show="sortColumn == 'creationTime' && sortReverse">
                <i class="fas fa-caret-up"></i>
              </span>
              <span v-show="sortColumn != 'creationTime'">
                <i class="fas fa-caret-up" style="visibility: hidden"></i>
              </span>
            </th>
            <th @click="updateSorting('updatedTime')" class="sortable">
              Updated on
              <span v-show="sortColumn == 'updatedTime' && !sortReverse">
                <i class="fas fa-caret-down"></i>
              </span>
              <span v-show="sortColumn == 'updatedTime' && sortReverse">
                <i class="fas fa-caret-up"></i>
              </span>
              <span v-show="sortColumn != 'updatedTime'">
                <i class="fas fa-caret-up" style="visibility: hidden"></i>
              </span>
            </th>
            <th @click="updateSorting('dataUploadTime')" class="sortable">
              Data uploaded on
              <span v-show="sortColumn == 'dataUploadTime' && !sortReverse">
                <i class="fas fa-caret-down"></i>
              </span>
              <span v-show="sortColumn == 'dataUploadTime' && sortReverse">
                <i class="fas fa-caret-up"></i>
              </span>
              <span v-show="sortColumn != 'dataUploadTime'">
                <i class="fas fa-caret-up" style="visibility: hidden"></i>
              </span>
            </th>
            <th>Actions</th>
            <th>Data spreadsheet</th>
            <th>Project file</th>
          </tr>
        </thead>
        <tbody>
          <tr v-for="projectSummary in sortedFilteredProjectSummaries" :class="{ highlighted: activeProject.uid === projectSummary.uid }">
            <td>
              <input type="checkbox" @click="uncheckSelectAll()" v-model="projectSummary.selected"/>
            </td>
            <td>{{ projectSummary.projectName }}</td>
            <td>
              <button class="btn __green" @click="openProject(projectSummary.uid)">Open</button>
            </td>
            <td>{{ projectSummary.creationTime }}</td>
            <td>{{ projectSummary.updateTime ? projectSummary.updateTime: 
              'No modification' }}</td>
            <td>{{ projectSummary.spreadsheetUploadTime ?  projectSummary.spreadsheetUploadTime: 
              'No data uploaded' }}</td>
            <td style="white-space: nowrap">
              <button class="btn" @click="copyProject(projectSummary.uid)">Copy</button>
              <button class="btn" @click="renameProject(projectSummary.uid)">Rename</button>
            </td>
            <td style="white-space: nowrap">
              <button class="btn" @click="uploadSpreadsheetToProject(projectSummary.uid)">Upload</button>
              <button class="btn" @click="downloadSpreadsheetFromProject(projectSummary.uid)">Download</button>
            </td>
            <td style="white-space: nowrap">
              <button class="btn" @click="downloadProjectFile(projectSummary.uid)">Download</button>
              <button class="btn" @click="downloadProjectFileWithResults(projectSummary.uid)">Download with results</button>
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
      // List of projects to choose from (by project name)
      demoProjectList: [],

      // Selected project (by name)
      selectedDemoProject: '',

      // Placeholder text for table filter box
      filterPlaceholder: '\u{1f50e} Filter Projects',

      // Text in the table filter box
      filterText: '',

      // Are all of the projects selected?
      allSelected: false, 

      // Column of table used for sorting the projects
      sortColumn: 'name',  // name, creationTime, updatedTime, dataUploadTime

      // Sort in reverse order?
      sortReverse: false, 

      // List of summary objects for projects the user has
      projectSummaries: [],

      // Active project
      activeProject: {},

      // List of project summaries
      demoProjectSummaries: 
        [
          {
            projectName: 'Concentrated (demo)', 
            creationTime: '2017-Sep-21 08:44 AM',
            updateTime: '2017-Sep-21 08:44 AM',
            spreadsheetUploadTime: '2017-Sep-21 08:44 AM',
            uid: 1,
            selected: false
          }, 
          {
            projectName: 'Generalized (demo)',
            creationTime: '2017-Sep-22 08:44 AM',
            updateTime: '',
            spreadsheetUploadTime: '2017-Sep-22 08:44 AM',
            uid: 2,
            selected: false
          },
          {
            projectName: 'Gaussian Graph 1', 
            creationTime: '2017-Sep-21 08:44 AM',
            updateTime: '2017-Sep-21 08:44 AM',
            spreadsheetUploadTime: '2017-Sep-21 08:44 AM',
            uid: 3,
            selected: false
          }, 
          {
            projectName: 'Gaussian Graph 2',
            creationTime: '2017-Sep-22 08:44 AM',
            updateTime: '',
            spreadsheetUploadTime: '2017-Sep-22 08:44 AM',
            uid: 4,
            selected: false
          }, 
          {
            projectName: 'Gaussian Graph 3',
            creationTime: '2017-Sep-23 08:44 AM',
            updateTime: '2017-Sep-23 08:44 AM',
            spreadsheetUploadTime: '',
            uid: 5,
            selected: false
          },
          {
            projectName: 'Uniform Graph 1', 
            creationTime: '2017-Mar-21 08:44 AM',
            updateTime: '2017-Mar-21 08:44 AM',
            spreadsheetUploadTime: '2017-Mar-21 08:44 AM',
            uid: 6,
            selected: false
          }, 
          {
            projectName: 'Uniform Graph 2',
            creationTime: '2017-Mar-22 08:44 AM',
            updateTime: '',
            spreadsheetUploadTime: '2017-Mar-22 08:44 AM',
            uid: 7,
            selected: false
          }, 
          {
            projectName: 'Uniform Graph 3',
            creationTime: '2017-Mar-23 08:44 AM',
            updateTime: '2017-Mar-23 08:44 AM',
            spreadsheetUploadTime: '',
            uid: 8,
            selected: false
          }
        ]
    }
  },

  computed: {
    sortedFilteredProjectSummaries() {
      return this.applyFilter(this.applySorting(this.projectSummaries))
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

      // Find the object in the default project summaries that matches what's 
      // selected in the select box.
      let foundProject = this.demoProjectSummaries.find(demoProj => 
        demoProj.projectName == this.selectedDemoProject)

      // Make a deep copy of the found object by JSON-stringifying the old 
      // object, and then parsing the result back into a new object.
      let newProject = JSON.parse(JSON.stringify(foundProject));

      // Push the deep copy to the projectSummaries list.
      this.projectSummaries.push(newProject)

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

    selectAll() {
      console.log('selectAll() called')

      // For each of the projects, set the selection of the project to the 
      // _opposite_ of the state of the all-select checkbox's state.
      // NOTE: This function depends on it getting called before the 
      // v-model state is updated.  If there are some cases of Vue 
      // implementation where these happen in the opposite order, then 
      // this will not give the desired result.
      this.projectSummaries.forEach(theProject => theProject.selected = !this.allSelected)
    },

    uncheckSelectAll() {
      this.allSelected = false
    },

    updateSorting(sortColumn) {
      console.log('updateSorting() called')

      // If the active sorting column is clicked...
      if (this.sortColumn === sortColumn) {
          // Reverse the sort.
          this.sortReverse = !this.sortReverse
      } 
      // Otherwise.
      else {
        // Select the new column for sorting.
        this.sortColumn = sortColumn

        // Set the sorting for non-reverse.
        this.sortReverse = false
      }
    },

    applyFilter(projects) {
      console.log('applyFilter() called')

      return projects.filter(theProject => theProject.projectName.toLowerCase().indexOf(this.filterText.toLowerCase()) !== -1)
    },

    applySorting(projects) {
      console.log('applySorting() called')

      return projects.sort((proj1, proj2) => 
        {
          let sortDir = this.sortReverse ? -1: 1
          if (this.sortColumn === 'name') {
            return (proj1.projectName > proj2.projectName ? sortDir: -sortDir)
          }
          else if (this.sortColumn === 'creationTime') {
            return proj1.creationTime > proj2.creationTime ? sortDir: -sortDir
          }
          else if (this.sortColumn === 'updatedTime') {
            return proj1.updateTime > proj2.updateTime ? sortDir: -sortDir
          }
          else if (this.sortColumn === 'dataUploadTime') {
            return proj1.spreadsheetUploadTime > proj2.spreadsheetUploadTime ? sortDir: -sortDir
          } 
        }
      )
    },

    openProject(uid) {
      // Find the project that matches the UID passed in.
      let matchProject = this.projectSummaries.find(theProj => theProj.uid === uid)

      console.log('openProject() called for ' + matchProject.projectName)

      // Set the active project to the matched project.
      this.activeProject = matchProject
    },

    copyProject(uid) {
      // Find the project that matches the UID passed in.
      let matchProject = this.projectSummaries.find(theProj => theProj.uid === uid)

      console.log('copyProject() called for ' + matchProject.projectName)
    },

    renameProject(uid) {
      // Find the project that matches the UID passed in.
      let matchProject = this.projectSummaries.find(theProj => theProj.uid === uid)

      console.log('renameProject() called for ' + matchProject.projectName)
    },

    uploadSpreadsheetToProject(uid) {
      // Find the project that matches the UID passed in.
      let matchProject = this.projectSummaries.find(theProj => theProj.uid === uid)

      console.log('uploadSpreadsheetToProject() called for ' + matchProject.projectName)
    },

    downloadSpreadsheetFromProject(uid) {
      // Find the project that matches the UID passed in.
      let matchProject = this.projectSummaries.find(theProj => theProj.uid === uid)

      console.log('downloadSpreadsheetFromProject() called for ' + matchProject.projectName)
    },

    downloadProjectFile(uid) {
      // Find the project that matches the UID passed in.
      let matchProject = this.projectSummaries.find(theProj => theProj.uid === uid)

      console.log('downloadProjectFile() called for ' + matchProject.projectName)
    },

    downloadProjectFileWithResults(uid) {
      // Find the project that matches the UID passed in.
      let matchProject = this.projectSummaries.find(theProj => theProj.uid === uid)

      console.log('downloadProjectFileWithResults() called for ' + matchProject.projectName)
    },

    deleteSelectedProjects() {
      // Pull out the names of the projects that are selected.
      let selectProjects = this.projectSummaries.filter(theProj => 
        theProj.selected).map(theProj => theProj.projectName)

      console.log('deleteSelectedProjects() called for ', selectProjects)
    },

    downloadSelectedProjects() {
      // Pull out the names of the projects that are selected.
      let selectProjects = this.projectSummaries.filter(theProj => 
        theProj.selected).map(theProj => theProj.projectName)

      console.log('downloadSelectedProjects() called for ', selectProjects)
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
