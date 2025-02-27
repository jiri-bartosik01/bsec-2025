<template>
    <div ref="mapContainer" class="map-container"></div>
</template>

<script setup>
import {onMounted, ref} from "vue";
import maplibregl from "maplibre-gl";

const emit = defineEmits(['MapClicked']);
const mapContainer = ref(null);
let map;


const switchMapSource = (source) => {
    const newSource = source === 'basic' ? 'basic-tiles' : 'aerial-tiles';
    map.setStyle({
        version: 8,
        sources: {
            [newSource]: {
                type: 'raster',
                url: `https://api.mapy.cz/v1/maptiles/${source}/tiles.json?apikey=D1D_iVhi-pbGrWFW80ijlZmC_HRQzZaUa59gV-7ZaXo`,
                tileSize: 256,
            },
        },
        layers: [{
            id: 'tiles',
            type: 'raster',
            source: newSource,
        }],
    });
};

defineExpose({
    switchMapSource,
});

onMounted(() => {
    const API_KEY = "D1D_iVhi-pbGrWFW80ijlZmC_HRQzZaUa59gV-7ZaXo";
     map = new maplibregl.Map({
        container: mapContainer.value, // Use the ref value here
        center: [16.608504478, 49.195213619],
        zoom: 15,
        attributionControl: false,
        style: {
            version: 8,
            sources: {
                'basic-tiles': {
                    type: 'raster',
                    url: `https://api.mapy.cz/v1/maptiles/basic/tiles.json?apikey=${API_KEY}`,
                    tileSize: 256,
                },
                'aerial-tiles': {
                    type: 'raster',
                    url: `https://api.mapy.cz/v1/maptiles/aerial/tiles.json?apikey=${API_KEY}`,
                    tileSize: 256,
                },
            },
            layers: [{
                id: 'tiles',
                type: 'raster',
                source: 'basic-tiles',
            }],
        },
    });
    map.addControl(new maplibregl.NavigationControl());

    class LogoControl {
        onAdd(map) {
            this._map = map;
            this._container = document.createElement('div');
            this._container.className = 'maplibregl-ctrl';
            this._container.innerHTML = '<a href="http://mapy.cz/" target="_blank"><img  width="100px" src="https://api.mapy.cz/img/api/logo.svg" ></a>';

            return this._container;
        }

        onRemove() {
            this._container.parentNode.removeChild(this._container);
            this._map = undefined;
        }
    }

    map.addControl(new LogoControl(), 'bottom-left');
    map.on('click', (e) => {
        emit('MapClicked', e);
    });

    // Request user's current location and center the map
    if (navigator.geolocation) {
        navigator.geolocation.getCurrentPosition((position) => {
            const userLocation = [position.coords.longitude, position.coords.latitude];
            map.setCenter(userLocation);
        }, (error) => {
            console.error("Error getting user's location: ", error);
        });
    } else {
        console.error("Geolocation is not supported by this browser.");
    }
});
</script>

<style>
.map-container {
    position: absolute;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    overflow-y: hidden;
    z-index: 0; /* Ensures map is behind everything */
}
</style>