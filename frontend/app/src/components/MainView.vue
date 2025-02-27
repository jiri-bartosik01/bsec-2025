<script setup>
import { ref } from "vue";
import MapView from "./MapView.vue";
import PlaceView from "./PlaceView.vue";

const showFilter = ref(false);
</script>

<template>
    <div class="relative w-screen h-screen">
        <!-- The Map MUST be before UI elements so it renders below -->
        <MapView />

        <!-- Floating Pill Navbar with 90% width -->
        <div
                class="absolute top-4 left-1/2 transform -translate-x-1/2 bg-base-100 shadow-lg backdrop-blur-md bg-opacity-80 w-95/100 max-w-3xl px-4 py-4 rounded-full flex items-center justify-between z-10"
        >
            <!-- Left Dropdown Button -->
            <div class="dropdown">
                <div tabindex="0" role="button" class="btn btn-ghost btn-circle" onclick="my_modal_3.showModal()">
                    <svg width="30px" height="30px" viewBox="0 0 24 24" fill="none"
                         xmlns="http://www.w3.org/2000/svg">
                        <circle cx="12" cy="12" r="10" stroke="#1C274C" stroke-width="1.5"></circle>
                        <path d="M12 17V11" stroke="#1C274C" stroke-width="1.5" stroke-linecap="round"></path>
                        <circle cx="1" cy="1" r="1" transform="matrix(1 0 0 -1 11 9)" fill="#1C274C"></circle>
                    </svg>
                </div>
                <ul tabindex="0" class="menu menu-sm dropdown-content bg-base-100 rounded-box z-[1] mt-3 w-52 p-2 shadow">
                    <li><a>O projektu</a></li>
                </ul>
            </div>

            <!-- Title -->
            <div class="flex-1 text-center text-lg font-semibold">Mapa nehod</div>

            <!-- Right Filter Button -->
            <button @click="showFilter = !showFilter" class="btn btn-ghost btn-circle">
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
                <input type="text" class="input" placeholder="My awesome page"/>
                <p class="fieldset-label">You can edit page title later on from settings</p>
            </fieldset>
            <fieldset class="fieldset">
                <legend class="fieldset-legend">Řidič</legend>
                <input type="text" class="input" placeholder="My awesome page"/>
                <p class="fieldset-label">You can edit page title later on from settings</p>
            </fieldset>
            <div class="card-actions justify-end">
                <button class="btn btn-ghost">Zrušit filtr</button>
                <button class="btn btn-primary">Filtrovat</button>
            </div>
        </div>

        <div sty

        <!-- Details Modal -->
        <PlaceView/>

    </div>
</template>
