<script setup>
    const emit = defineEmits(['close']);
    const props = defineProps({
        place: {
            type: Object,
            required: true
        }
    });
</script>

<template>
    <div class=" z-5 card card-border bg-base-100 w-96 absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2">
        <div class="card-body">
            <h2 class="card-title">🚗💥</h2>
            <p>{{place.lat}}, {{place.lng}}</p>
            <div class="card-actions justify-end">
                <button class="btn btn-ghost" @click="emit('close')">Zavřít</button>
            </div>
        </div>
    </div>
</template>

<style scoped>

</style>