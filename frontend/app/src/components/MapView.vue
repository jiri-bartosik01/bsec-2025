<template>
    <div ref="mapContainer" class="map-container"></div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import maplibregl from "maplibre-gl";

const mapContainer = ref(null);

onMounted(() => {
    const map = new maplibregl.Map({
        container: mapContainer.value,
        style: "https://demotiles.maplibre.org/style.json", // Free MapLibre tile server
        center: [10, 50], // Default location
        zoom: 5,
    });

    // Add a heatmap layer (example)
    map.on("load", () => {
        map.addSource("heatmap-source", {
            type: "geojson",
            data: "your-geojson-url-or-data",
        });

        map.addLayer({
            id: "heatmap-layer",
            type: "heatmap",
            source: "heatmap-source",
            paint: {
                "heatmap-weight": ["interpolate", ["linear"], ["get", "value"], 0, 0, 100, 1],
                "heatmap-intensity": 1,
                "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"], 0, "blue", 1, "red"],
            },
        });
    });
});
</script>

<style>
.map-container {
    position: absolute;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    z-index: 0; /* Ensures map is behind everything */
}
</style>
