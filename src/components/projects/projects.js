import React from "react"
import { arrayOf, ProjectType, shape } from "../../types"
import Project from "./project"

const Projects = ({ projects }) => (
  <>
    <h5 className="font-header font-semibold text-front text-lg uppercase mb-3">
      Projects and teams
    </h5>
    {projects.map((project, i) => (
      <Project key={`${project.name}_${i}`} {...project} />
    ))}
  </>
)

Projects.propTypes = {
  projects: arrayOf(shape(ProjectType)),
}

export default Projects