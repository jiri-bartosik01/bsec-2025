<template>
    <div ref="mapContainer" class="map-container"></div>
</template>

<script setup>
import {onMounted, ref} from "vue";
import maplibregl from "maplibre-gl";

const emit = defineEmits(['MapClicked', 'mapDataLoaded', 'mapDataLoading']);
const mapContainer = ref(null);
let map;

const fetchCentroids = async () => {
    emit('mapDataLoading');
    const url = 'http://10.147.19.154:8000/points';
    const response = await fetch(url);
    const data = await response.json();
    processPoints(data);
    emit('mapDataLoaded');
};

const processPoints = (points) => {
    const clusters = points.reduce((acc, point) => {
        if (!acc[point.cluster_id]) {
            acc[point.cluster_id] = [];
        }
        acc[point.cluster_id].push([point.x, point.y]);
        return acc;
    }, {});

    Object.values(clusters).forEach(cluster => {
        const polygon = createPolygon(cluster);
        addPolygonToMap(polygon);
    });
};

const createPolygon = (points) => {
    // Assuming points are in the correct order to form a polygon
    return {
        type: 'Feature',
        geometry: {
            type: 'Polygon',
            coordinates: [points]
        }
    };
};

const addPolygonToMap = (polygon) => {
    map.addSource(`polygon-${Math.random()}`, {
        type: 'geojson',
        data: polygon
    });

    map.addLayer({
        id: `polygon-${Math.random()}`,
        type: 'fill',
        source: `polygon-${Math.random()}`,
        layout: {},
        paint: {
            'fill-color': '#088',
            'fill-opacity': 0.8
        }
    });
};

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
    fetchCentroids();

    const API_KEY = "D1D_iVhi-pbGrWFW80ijlZmC_HRQzZaUa59gV-7ZaXo";
    map = new maplibregl.Map({
        container: mapContainer.value,
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