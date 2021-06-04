module.exports = {
  siteMetadata: {
    description: "Personal website of Sergios Karagiannakos",
    locale: "en",
    showThemeLogo: false,
    title: "Sergios Karagiannakos",
    formspreeEndpoint: "https://formspree.io/sergioskarag@gmail.com}",
  },
  plugins: [
    {
      resolve: `gatsby-plugin-postcss`,
      options: {
        postCssPlugins: [
          require("tailwindcss")(require("./tailwind.config")("bright-green")), //the theme can be changed https://github.com/SergiosKar/gatsby-theme-intro
          require("postcss-input-range"),
          require("autoprefixer"),
        ],
      },
    },
    `gatsby-plugin-react-helmet`,
    `gatsby-transformer-yaml`,
    {
      resolve: `gatsby-source-filesystem`,
      options: {
        path: "content/",
      },
    },
    `gatsby-plugin-react-svg`,
    `gatsby-plugin-image`,
    `gatsby-transformer-sharp`,
    `gatsby-plugin-sharp`,
  ],
}
