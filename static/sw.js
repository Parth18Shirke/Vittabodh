/**
 * VittaBodh Service Worker — cache-first for static assets.
 * Provides basic offline capability: the app shell loads from cache
 * even without a network, while API calls always hit the network.
 */

const CACHE_NAME = "vittabodh-v1";
const STATIC_ASSETS = [
  "/static/css/style.css",
  "/static/manifest.json",
  "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap",
];

// Install: pre-cache static assets
self.addEventListener("install", (event) => {
  self.skipWaiting();
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(STATIC_ASSETS).catch(() => {}))
  );
});

// Activate: clean up old caches
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k)))
    )
  );
  self.clients.claim();
});

// Fetch: network-first for API/navigation, cache-first for static assets
self.addEventListener("fetch", (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Always go network for API calls, form submissions, auth
  if (url.pathname.startsWith("/api/") || request.method !== "GET") {
    return;
  }

  // Cache-first for static assets (CSS, fonts, manifest)
  if (url.pathname.startsWith("/static/") || url.hostname === "fonts.googleapis.com") {
    event.respondWith(
      caches.match(request).then(
        (cached) => cached || fetch(request).then((resp) => {
          if (resp.ok) {
            const clone = resp.clone();
            caches.open(CACHE_NAME).then((c) => c.put(request, clone));
          }
          return resp;
        })
      )
    );
    return;
  }

  // Network-first for HTML pages
  event.respondWith(
    fetch(request).catch(() => caches.match(request))
  );
});
