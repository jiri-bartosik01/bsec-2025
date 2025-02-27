<script setup>
import { ref } from "vue";
import MapView from "./MapView.vue";
import PlaceView from "./PlaceView.vue";

const showFilter = ref(false);
const showPlaceView = ref(false);
const focusedPlace = ref({"lat": 0, "lng": 0});
const mapViewRef = ref(null);

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
        <MapView ref="mapViewRef" @MapClicked="handleMapClicked"/>

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
            <div class="flex-1 text-center text-lg font-semibold">Mapa nehod</div>

            <!-- Right Filter Button -->
            <button @click="toggleFilterVisibility" class="btn btn-ghost btn-circle">
                <svg width="30px" height="30px" viewBox="0 0 1024 1024" class="icon"
                     xmlns="http://www.w3.org/2000/svg" fill="#000000">
                    <path fill="#000000"
                          d="M384 523.392V928a32 32 0 0046.336 28.608l192-96A32 32 0 00640 832V523.392l280.768-343.104a32 32 0 10-49.536-40.576l-288 352A32 32 0 00576 512v300.224l-128 64V512a32 32 0 00-7.232-20.288L195.52 192H704a32 32 0 100-64H128a32 32 0 00-24.768 52.288L384 523.392z"></path>
                </svg>
            </button>
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
        <PlaceView v-if="showPlaceView" @close="togglePlaceViewVisibility" class="z-30" :place="focusedPlace"/>

    </div>
</template>
