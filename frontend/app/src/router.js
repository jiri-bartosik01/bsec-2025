import { createRouter, createWebHistory } from "vue-router";
import MapView from "./components/MapView.vue";
import FilterView from "./components/FilterView.vue";
import PlaceView from "./components/PlaceView.vue";
import MainView from "./components/MainView.vue";

// Define routes
const routes = [
    { path: "/", component: MainView },
];

// Create router instance
const router = createRouter({
    history: createWebHistory(),
    routes,
});

export default router;
