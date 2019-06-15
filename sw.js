/**
 * Copyright 2016 Google Inc. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
*/

// DO NOT EDIT THIS GENERATED OUTPUT DIRECTLY!
// This file should be overwritten as part of your build process.
// If you need to extend the behavior of the generated service worker, the best approach is to write
// additional code and include it using the importScripts option:
//   https://github.com/GoogleChrome/sw-precache#importscripts-arraystring
//
// Alternatively, it's possible to make changes to the underlying template file and then use that as the
// new base for generating output, via the templateFilePath option:
//   https://github.com/GoogleChrome/sw-precache#templatefilepath-string
//
// If you go that route, make sure that whenever you update your sw-precache dependency, you reconcile any
// changes made to this original template file with your modified copy.

// This generated service worker JavaScript will precache your site's resources.
// The code needs to be saved in a .js file at the top-level of your site, and registered
// from your pages in order to be used. See
// https://github.com/googlechrome/sw-precache/blob/master/demo/app/js/service-worker-registration.js
// for an example of how you can register this script and handle various service worker events.

/* eslint-env worker, serviceworker */
/* eslint-disable indent, no-unused-vars, no-multiple-empty-lines, max-nested-callbacks, space-before-function-paren, quotes, comma-spacing */
'use strict';

var precacheConfig = [["/404.html","545f02c842ea95e169c7bcf697df651b"],["/Actor_critics/index.html","98ffc86a303619144274f94a0c94788e"],["/Autoencoder/index.html","e64704a2541dfd3696ede5a1f217bea4"],["/Bitcon_prediction_LSTM/index.html","c91aff7551f73ab45ca90960d699e314"],["/Deep_Q_Learning/index.html","b88422a4d998be6f0a00d2e39253cf4e"],["/Deep_learning/index.html","a2ac990900d2ace661429904174d5b28"],["/Document_clustering/index.html","f40f1451ced930058a993b50483be97f"],["/Generative_Artificial_Intelligence/index.html","7ce877d11a06679bc15532a1d5051011"],["/Localization_and_Object_Detection/index.html","4d006bec1a85e72f2fc149c6108e21cf"],["/NALU/index.html","b6a47218ae76e2763c53311436cf1a1d"],["/Neural_Network_from_scratch/index.html","c714f317b55cf2f5173e80409cb5a523"],["/Neural_Network_from_scratch_part2/index.html","1b82be171f5ae155915e253042b255cf"],["/Policy-Gradients/index.html","204851ba4a0e190a519b9061703c782d"],["/Reinforcement_learning/index.html","e3eb475a0d4505ea61788312e4ee6f98"],["/Self_driving_cars/index.html","678ca7b789251d70eeed2cfec318475a"],["/Semantic_Segmentation/index.html","c2bce05d6b24bbaa81b9e6e83e9c6731"],["/TRPO_PPO/index.html","6a8d0b8149d49f227a97f270d9d37328"],["/Taking_Deep_Q_Networks_a_step_further/index.html","0789e128aa2f0159e0d1e4b442e7d26a"],["/YOLO/index.html","6372f46b3e367b66d4d28bfe2c7f6cd4"],["/about/index.html","11b35030cc6ca886d4587cf27c60a97b"],["/assets/css/main.css","f4c4e944bc8b618bbabc264965381b39"],["/assets/img/background_img2.jpg","89e4f8ffd77c4c4c5ace1c134d794e75"],["/assets/img/favicon.jpg","ffb9f5c8afdda7fa4f3fd697e5147182"],["/assets/img/icons/android-chrome-192x192.png","4df4c8779d47bcaa69516050281773b9"],["/assets/img/icons/android-chrome-256x256.png","939ec88a61f407945a27d867fca1651d"],["/assets/img/icons/apple-touch-icon.png","366666899d15cf8f6811cc73ee0d63de"],["/assets/img/icons/favicon-16x16.png","f625044491b20a5df78571ba266cbcf6"],["/assets/img/icons/favicon-32x32.png","67502381e45848a4ab76123364675ffe"],["/assets/img/icons/icon-github.svg","4e06335104a29f91e08d4ef420da7679"],["/assets/img/icons/icon-instagram.svg","1e1119e2628235ee4c8771bff15eb2ca"],["/assets/img/icons/icon-twitter.svg","30551913d5399d6520e8a74b6f1e23f0"],["/assets/img/icons/mstile-150x150.png","1cea2ceb806d1a018330a51a1d8b73b6"],["/assets/img/icons/safari-pinned-tab.svg","398ef6b25c0f7f3f6e54c112a8facc5f"],["/assets/img/portofolio/chill.jpg","01061366ffd20f47364709a4a8990472"],["/assets/img/portofolio/code_analysis_image.jpg","4a125d88d3b7f1aa757ced1278be8d96"],["/assets/img/portofolio/etf_est_db.jpg","5fcfa79d21e604f43e1084855ee341b7"],["/assets/img/portofolio/mnist.jpg","2c4975223a9e0ec060f8420f767164b8"],["/assets/img/portofolio/robot_motion_planning.jpg","7edce12de9cb6843d484668e69f8f128"],["/assets/img/posts/Cartpool.jpg","54e28a4c3c77937182a5da4463992e7b"],["/assets/img/posts/DDQN.jpg","2d1b5e8be16480f00f8a13f178720d75"],["/assets/img/posts/DQN.jpg","6923b77260c7b9ac650998e6c016ea67"],["/assets/img/posts/DRL.jpg","7c4eb49ebec6e7065d988bdbb14996b8"],["/assets/img/posts/NAC.jpg","b25af88a4fbc916a4b070f29c4f7d5e5"],["/assets/img/posts/NALU_equat.jpg","2bd05168a5e97d47c667fed1d6b803bb"],["/assets/img/posts/NALU_img.jpg","0d3b93110077af4f98f1207aeea899fa"],["/assets/img/posts/NN.jpg","a69309810b1e3a63c6c2b53e4e1a5baf"],["/assets/img/posts/PER.jpg","7d2ab5b3a60c17cdad282193c1c1156b"],["/assets/img/posts/QValue.jpg","37c029f45ae960956b4af209e9733cdc"],["/assets/img/posts/RL.jpg","ac7d43ccea9250f87d19a0758ba71699"],["/assets/img/posts/RL_algortihms.jpg","e21ce9c09924bc7e43d87b9206717e4b"],["/assets/img/posts/RL_reward.jpg","6539cebae3e89d2140bfa48902c04a49"],["/assets/img/posts/RL_value.jpg","304e9860a111a704c5114540d0a2e800"],["/assets/img/posts/TDError.jpg","0f779e4dad5f867ed3a53b1691fffe36"],["/assets/img/posts/a2c.jpg","f1439795ad5e6022378b2cb47a56a45b"],["/assets/img/posts/a3c.jpg","a3548f07bf6cb4654b119a139b9edef3"],["/assets/img/posts/ac.jpg","84d6a927bf16c5b5a8fbd862db92d3d1"],["/assets/img/posts/autoencoder.jpg","0ba67a8b5ff61099f5de3544b003a445"],["/assets/img/posts/bellman.jpg","d02ac285a2fc0fb7e721c1e8a98fd691"],["/assets/img/posts/bpa_equat.jpg","70bb7d6ce7cff1def7f9fcaa6920f478"],["/assets/img/posts/btc_prediction_plot.jpg","547e4d9e825b6aeb793ab54235b79b99"],["/assets/img/posts/conv_bpa.jpg","6ad168de4c784f2560a7a376c659bb4f"],["/assets/img/posts/conv_bpa_deltas.jpg","6aaf3b7a23c8e25971afb27a70b092fd"],["/assets/img/posts/cv_tasks.jpg","323f1629529ffd1a6202c5f7482d5975"],["/assets/img/posts/deep_dream.jpg","e5580963fdfb998bfe2103f4cbf5aa8c"],["/assets/img/posts/deep_learning.jpg","cbeef17e70974d0e2d14dcb860ac294c"],["/assets/img/posts/fasterrcnn.jpg","af0a27429eb557d460243dc31ca30cae"],["/assets/img/posts/fastrcnn.jpg","38b024640ee848989262adb74a8ffcca"],["/assets/img/posts/fcn1.jpg","8fbb16839ab4dbbe444826eea02ddad5"],["/assets/img/posts/fcn2.jpg","559e217e28d3b143d8c2e436aa29530b"],["/assets/img/posts/gan.jpg","a4126bf23fc73760960986ed2151e356"],["/assets/img/posts/gan_training.jpg","062fac10a334d2809f0baa26b84920f0"],["/assets/img/posts/lstm_cll.jpg","570225e9f8a751b09c0a97a0a86faf0a"],["/assets/img/posts/lstm_equations.jpg","69b196385b423311f07b15016939680a"],["/assets/img/posts/neuron.jpg","2559b89b7ddd192d7e771b0087f1f840"],["/assets/img/posts/nlp_20_0.jpg","0a389613a2fe3aabccb58012be3bae22"],["/assets/img/posts/nn_range.jpg","8190951fbe92314faab16c0f511ffc60"],["/assets/img/posts/pg_theorem.jpg","7230c247dd4469c43ae079d03ad187e3"],["/assets/img/posts/policy_gradient.jpg","6f34b7c17aa66671bca6426ae9946f32"],["/assets/img/posts/pong_pg.jpg","0bcbdcc7100593807a480aa67c8b407e"],["/assets/img/posts/ppo.jpg","fa8506db7389283c62b493b45e7ad0f8"],["/assets/img/posts/ppo_penalty.jpg","bba190626a1131ef28ddeb2f30cdf3a9"],["/assets/img/posts/ppo_trpo.jpg","dbd182979b8a07dc84483f5dafbf460b"],["/assets/img/posts/rcnn.jpg","2e656d944b757ff500cb11f13c76c436"],["/assets/img/posts/regions_proposals.jpg","55216c41b3bedc9287c00e1ffd53e196"],["/assets/img/posts/sdc_sensors.jpg","e614ca09900a650b4b7e66340ec70a65"],["/assets/img/posts/sdc_simulator.jpg","6967b7db00303d74f9192ace08dd8458"],["/assets/img/posts/semseg.jpg","36b49076a9482203139f8e0c0176818e"],["/assets/img/posts/trpo_eq.jpg","5c5bd9f6fa9a9d21db03f065e6df349d"],["/assets/img/posts/unet.jpg","80fd39f06125b2a1c62f4f35825a2a7b"],["/assets/img/posts/vae.jpg","bd5f65600f84bfc679bc2d9399fafd1f"],["/assets/img/posts/vae_mnist_results.jpg","2543a48e2b4293024237976959d0d4fc"],["/assets/img/posts/yolo.jpg","be7dac1824d008c605d4c92a99b767cb"],["/assets/img/posts/yolo_app.jpg","d3d6ef32de303b62c0c4a39ef3d75e42"],["/assets/img/posts/yolo_architecture.jpg","14acce3e358e0bee8013c882bb0bcd62"],["/assets/img/website_photo2.jpg","a73075e87b3d9512fcd86333e9b1db76"],["/assets/img/website_photo3.jpg","088647f8ba97db24d1d241e9e2197c88"],["/assets/js/bundle.js","b3ef060714dd10b3ea058172b49ca733"],["/contact/index.html","8e181829079d447c50fff5808c1c5a9f"],["/gulpfile.babel.js","59db5b5e8f59b09ab5b06fdb24d37743"],["/index.html","b74979e242f14140a85a3e5c2faf0dfa"],["/portofolio/Chill/index.html","66b9cdcc4e8715a9a1df81a11cc29398"],["/portofolio/Est_db/index.html","02a9b461e6c7c2e27227ff2ce06c2ea9"],["/portofolio/index.html","c92fa41a0ab2eed4ca7199d69db4e116"],["/portofolio/neural_network_images/index.html","6208747b30dea8386e51b732e2c4bf33"],["/portofolio/robot_motion_planning/index.html","ecc9438cb822f51eefc2647d07da5e3a"],["/portofolio/source_code_analysis/index.html","ab81798de649673cb6f2fd745a76123d"],["/sw.js","42ed9a8ea881a5674df82b5e8c27c9c4"]];
var cacheName = 'sw-precache-v3--' + (self.registration ? self.registration.scope : '');


var ignoreUrlParametersMatching = [/^utm_/];



var addDirectoryIndex = function(originalUrl, index) {
    var url = new URL(originalUrl);
    if (url.pathname.slice(-1) === '/') {
      url.pathname += index;
    }
    return url.toString();
  };

var cleanResponse = function(originalResponse) {
    // If this is not a redirected response, then we don't have to do anything.
    if (!originalResponse.redirected) {
      return Promise.resolve(originalResponse);
    }

    // Firefox 50 and below doesn't support the Response.body stream, so we may
    // need to read the entire body to memory as a Blob.
    var bodyPromise = 'body' in originalResponse ?
      Promise.resolve(originalResponse.body) :
      originalResponse.blob();

    return bodyPromise.then(function(body) {
      // new Response() is happy when passed either a stream or a Blob.
      return new Response(body, {
        headers: originalResponse.headers,
        status: originalResponse.status,
        statusText: originalResponse.statusText
      });
    });
  };

var createCacheKey = function(originalUrl, paramName, paramValue,
                           dontCacheBustUrlsMatching) {
    // Create a new URL object to avoid modifying originalUrl.
    var url = new URL(originalUrl);

    // If dontCacheBustUrlsMatching is not set, or if we don't have a match,
    // then add in the extra cache-busting URL parameter.
    if (!dontCacheBustUrlsMatching ||
        !(url.pathname.match(dontCacheBustUrlsMatching))) {
      url.search += (url.search ? '&' : '') +
        encodeURIComponent(paramName) + '=' + encodeURIComponent(paramValue);
    }

    return url.toString();
  };

var isPathWhitelisted = function(whitelist, absoluteUrlString) {
    // If the whitelist is empty, then consider all URLs to be whitelisted.
    if (whitelist.length === 0) {
      return true;
    }

    // Otherwise compare each path regex to the path of the URL passed in.
    var path = (new URL(absoluteUrlString)).pathname;
    return whitelist.some(function(whitelistedPathRegex) {
      return path.match(whitelistedPathRegex);
    });
  };

var stripIgnoredUrlParameters = function(originalUrl,
    ignoreUrlParametersMatching) {
    var url = new URL(originalUrl);
    // Remove the hash; see https://github.com/GoogleChrome/sw-precache/issues/290
    url.hash = '';

    url.search = url.search.slice(1) // Exclude initial '?'
      .split('&') // Split into an array of 'key=value' strings
      .map(function(kv) {
        return kv.split('='); // Split each 'key=value' string into a [key, value] array
      })
      .filter(function(kv) {
        return ignoreUrlParametersMatching.every(function(ignoredRegex) {
          return !ignoredRegex.test(kv[0]); // Return true iff the key doesn't match any of the regexes.
        });
      })
      .map(function(kv) {
        return kv.join('='); // Join each [key, value] array into a 'key=value' string
      })
      .join('&'); // Join the array of 'key=value' strings into a string with '&' in between each

    return url.toString();
  };


var hashParamName = '_sw-precache';
var urlsToCacheKeys = new Map(
  precacheConfig.map(function(item) {
    var relativeUrl = item[0];
    var hash = item[1];
    var absoluteUrl = new URL(relativeUrl, self.location);
    var cacheKey = createCacheKey(absoluteUrl, hashParamName, hash, false);
    return [absoluteUrl.toString(), cacheKey];
  })
);

function setOfCachedUrls(cache) {
  return cache.keys().then(function(requests) {
    return requests.map(function(request) {
      return request.url;
    });
  }).then(function(urls) {
    return new Set(urls);
  });
}

self.addEventListener('install', function(event) {
  event.waitUntil(
    caches.open(cacheName).then(function(cache) {
      return setOfCachedUrls(cache).then(function(cachedUrls) {
        return Promise.all(
          Array.from(urlsToCacheKeys.values()).map(function(cacheKey) {
            // If we don't have a key matching url in the cache already, add it.
            if (!cachedUrls.has(cacheKey)) {
              var request = new Request(cacheKey, {credentials: 'same-origin'});
              return fetch(request).then(function(response) {
                // Bail out of installation unless we get back a 200 OK for
                // every request.
                if (!response.ok) {
                  throw new Error('Request for ' + cacheKey + ' returned a ' +
                    'response with status ' + response.status);
                }

                return cleanResponse(response).then(function(responseToCache) {
                  return cache.put(cacheKey, responseToCache);
                });
              });
            }
          })
        );
      });
    }).then(function() {
      
      // Force the SW to transition from installing -> active state
      return self.skipWaiting();
      
    })
  );
});

self.addEventListener('activate', function(event) {
  var setOfExpectedUrls = new Set(urlsToCacheKeys.values());

  event.waitUntil(
    caches.open(cacheName).then(function(cache) {
      return cache.keys().then(function(existingRequests) {
        return Promise.all(
          existingRequests.map(function(existingRequest) {
            if (!setOfExpectedUrls.has(existingRequest.url)) {
              return cache.delete(existingRequest);
            }
          })
        );
      });
    }).then(function() {
      
      return self.clients.claim();
      
    })
  );
});


self.addEventListener('fetch', function(event) {
  if (event.request.method === 'GET') {
    // Should we call event.respondWith() inside this fetch event handler?
    // This needs to be determined synchronously, which will give other fetch
    // handlers a chance to handle the request if need be.
    var shouldRespond;

    // First, remove all the ignored parameters and hash fragment, and see if we
    // have that URL in our cache. If so, great! shouldRespond will be true.
    var url = stripIgnoredUrlParameters(event.request.url, ignoreUrlParametersMatching);
    shouldRespond = urlsToCacheKeys.has(url);

    // If shouldRespond is false, check again, this time with 'index.html'
    // (or whatever the directoryIndex option is set to) at the end.
    var directoryIndex = 'index.html';
    if (!shouldRespond && directoryIndex) {
      url = addDirectoryIndex(url, directoryIndex);
      shouldRespond = urlsToCacheKeys.has(url);
    }

    // If shouldRespond is still false, check to see if this is a navigation
    // request, and if so, whether the URL matches navigateFallbackWhitelist.
    var navigateFallback = '';
    if (!shouldRespond &&
        navigateFallback &&
        (event.request.mode === 'navigate') &&
        isPathWhitelisted([], event.request.url)) {
      url = new URL(navigateFallback, self.location).toString();
      shouldRespond = urlsToCacheKeys.has(url);
    }

    // If shouldRespond was set to true at any point, then call
    // event.respondWith(), using the appropriate cache key.
    if (shouldRespond) {
      event.respondWith(
        caches.open(cacheName).then(function(cache) {
          return cache.match(urlsToCacheKeys.get(url)).then(function(response) {
            if (response) {
              return response;
            }
            throw Error('The cached response that was expected is missing.');
          });
        }).catch(function(e) {
          // Fall back to just fetch()ing the request if some unexpected error
          // prevented the cached response from being valid.
          console.warn('Couldn\'t serve response for "%s" from cache: %O', event.request.url, e);
          return fetch(event.request);
        })
      );
    }
  }
});







