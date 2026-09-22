/* Offline shell for the path-tracking demo.
 *
 * The demo has no build step and no third-party requests, so caching the eight
 * entries below is enough to make it work with no network at all.
 */
const CACHE = "rne-path-tracking-v2";
const ASSETS = [
  "./",
  "./index.html",
  "./app.js",
  "./engine.js",
  "./manifest.json",
  "./data/reference.json",
  "./icons/icon-192.png",
  "./icons/icon-512.png"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(ASSETS)).then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  if (event.request.method !== "GET") {
    return;
  }
  event.respondWith(
    caches.match(event.request).then((hit) => {
      if (hit) {
        return hit;
      }
      return fetch(event.request).then((response) => {
        const copy = response.clone();
        caches.open(CACHE).then((cache) => cache.put(event.request, copy)).catch(() => {});
        return response;
      });
    }).catch(() => caches.match("./index.html"))
  );
});
