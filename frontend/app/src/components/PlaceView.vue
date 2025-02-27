<script setup>
import {onMounted, ref} from "vue";

    const emit = defineEmits(['close']);
    const props = defineProps({
        place: {
            type: Object,
            required: true
        }
    });

    const isLoading = ref(true);

    const fetchPlaceData = async () => {
        // Fetch place data from API
        isLoading.value = true;
        setTimeout(() => {
            isLoading.value = false;
        }, 2000);
    };

    onMounted(() => {
        fetchPlaceData();
    });
</script>

<template>
    <div class=" z-5 card card-border bg-base-100 w-96 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
        <div class="card-body">
            <div style="height: 32px">
                <div v-if="isLoading">
                    <span class="loading loading-ring loading-xl"></span>
                </div>
                <div v-else>
                    <h2 class="card-title">🚗💥</h2>
                </div>
            </div>
            <div v-if="!isLoading">
                <p>{{place.lat}}, {{place.lng}}</p>
            </div>
            <div v-else>
                <div class="skeleton h-4 w-30"></div>
<!--                <div class="skeleton h-4 w-30 mt-3"></div>-->
<!--                <div class="skeleton h-4 w-30 mt-3"></div>-->
            </div>
            <div class="card-actions justify-end">
                <button class="btn btn-ghost" @click="emit('close')">Zavřít</button>
            </div>
        </div>
    </div>
</template>

<style scoped>

</style>