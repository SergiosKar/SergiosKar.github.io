import React from "react"
import { arrayOf, ProfileType, shape, SocialType } from "../../types"
import SocialLinks from "../social-links/social-links"
import Location from "./location"
import ProfileImage from "./profile-image"

const Sidebar = ({ profile, social }) => (
  <aside className="w-full lg:w-1/3 lg:border-r border-line lg:px-6 xl:px-12">
    <div className="flex flex-col h-full ">
      <div>
        <h2 className="font-header font-light text-front text-2xl leading-none mb-4">
          {profile.profession}
        </h2>
        <h1 className="font-header font-black text-front text-4xl leading-none break-words mb-6">
          {profile.name}
        </h1>
        {profile.image && (
          <ProfileImage image={profile.image} name={profile.name} />
        )}
        <br />
        {profile.location && (
          <Location
            location={profile.location}
            relocation={profile.relocation}
          />
        )}
      </div>

      <div className="pt-8 pb-12 lg:py-12">
        <h5 className="font-header font-semibold text-front text-sm uppercase mb-3">
          Connect
        </h5>
        <SocialLinks social={social} />
        <br />
        {profile.scholar && (
          <a className="no-underline hover:underline" href={profile.scholar}>
            Google scholar
          </a>
        )}
        <br />
        {profile.email && (
          <a
            className="no-underline hover:underline"
            href={`mailto:${profile.email}`}
          >
            {profile.email}
          </a>
        )}
      </div>
    </div>
  </aside>
)

Sidebar.propTypes = {
  profile: shape(ProfileType),
  social: arrayOf(shape(SocialType)),
}

export default Sidebar