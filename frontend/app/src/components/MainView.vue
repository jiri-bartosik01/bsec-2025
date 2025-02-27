<script setup>
import { ref } from "vue";
import MapView from "./MapView.vue";
import PlaceView from "./PlaceView.vue";

const showFilter = ref(false);
const showPlaceView = ref(false);
const focusedPlace = ref({"lat": 0, "lng": 0});
const mapViewRef = ref(null);
const isLoading = ref(true);

const toggleFilterVisibility = () => {
    if (!showPlaceView.value) {
        showFilter.value = !showFilter.value;
    }
};

const togglePlaceViewVisibility = () => {
    showPlaceView.value = !showPlaceView.value;
};

const handleMapClicked = (event) => {
    console.log("Map clicked at: ", event.lngLat);
    focusedPlace.value = event.lngLat;
    showPlaceView.value = true;
};

const switchToBasicMap = () => {
    mapViewRef.value.switchMapSource('basic');
};

const switchToAerialMap = () => {
    mapViewRef.value.switchMapSource('aerial');
};
</script>

<template>
    <div class="relative w-screen h-screen">
        <!-- The Map MUST be before UI elements so it renders below -->
        <MapView ref="mapViewRef" @MapClicked="handleMapClicked" @mapDataLoaded="isLoading = false" @mapDataLoading="isLoading = true"/>

        <!-- Floating Pill Navbar with 90% width -->
        <div
                class="absolute top-4 left-1/2 transform -translate-x-1/2 bg-base-100 shadow-lg backdrop-blur-md bg-opacity-80 w-95/100 max-w-3xl px-4 py-4 rounded-full flex items-center justify-between z-10"
        >
            <!-- Left Dropdown Button -->
            <div class="dropdown">
                <div tabindex="0" role="button" class="btn btn-ghost btn-circle" onclick="my_modal_3.showModal()">
                    <svg width="64px" height="64px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round" stroke="#CCCCCC" stroke-width="0.048"></g><g id="SVGRepo_iconCarrier"> <g id="Edit / Layer"> <path id="Vector" d="M21 14L12 20L3 14M21 10L12 16L3 10L12 4L21 10Z" stroke="#000000" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"></path> </g> </g></svg>                </div>
                <ul tabindex="0" class="menu menu-sm dropdown-content bg-base-100 rounded-box z-[1] mt-3 w-52 p-2 shadow">
                    <li @click="switchToBasicMap"><a>Základní mapa</a></li>
                    <li @click="switchToAerialMap"><a>Satelitní mapa</a></li>
                </ul>
            </div>

            <!-- Title -->
            <div class="flex-1 text-center text-xl font-semibold">Riziková místa v Brně a okolí</div>

            <!-- Right Filter Button -->
            <div>
                <span class="loading loading-ring loading-lg mr-1" v-if="isLoading"></span>
                <div v-else>
                    <svg width="36px" height="36px" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"><g id="SVGRepo_bgCarrier" stroke-width="0"></g><g id="SVGRepo_tracerCarrier" stroke-linecap="round" stroke-linejoin="round"></g><g id="SVGRepo_iconCarrier"> <path d="M8.5 12.5L10.5 14.5L15.5 9.5" stroke="#1C274C" stroke-width="1.2" stroke-linecap="round" stroke-linejoin="round"></path> <path d="M7 3.33782C8.47087 2.48697 10.1786 2 12 2C17.5228 2 22 6.47715 22 12C22 17.5228 17.5228 22 12 22C6.47715 22 2 17.5228 2 12C2 10.1786 2.48697 8.47087 3.33782 7" stroke="#1C274C" stroke-width="1.2" stroke-linecap="round"></path> </g></svg>
                </div>
            </div>
        </div>

        <!-- Floating Filter UI -->
        <div v-show="showFilter"
             class="absolute right-4 top-30 bg-base-100 w-96 rounded-lg shadow-lg p-4">
            <h2 class="card-title">Filtrovat data</h2>
            <fieldset class="fieldset">
                <legend class="fieldset-legend">Povětrnostní podmínky</legend>
                <input type="text" class="input" placeholder=""/>
            </fieldset>
            <fieldset class="fieldset">
                <legend class="fieldset-legend">Řidič</legend>
                <input type="text" class="input" placeholder=""/>
            </fieldset>
            <div class="card-actions justify-end">
                <button class="btn btn-ghost" @click="toggleFilterVisibility">Zrušit filtr</button>
                <button class="btn btn-primary">Filtrovat</button>
            </div>
        </div>

        <div v-show="showPlaceView" class="fixed inset-0 backdrop-blur-md bg-opacity-30 z-20 w-screen h-screen"></div>

        <!-- Details Modal -->
        <PlaceView v-if="showPlaceView" @close="togglePlaceViewVisibility" class="z-30 shadow-lg" :place="focusedPlace"/>

    </div>
</template>
