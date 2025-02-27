<script setup>

</script>

<template>
    <div style="margin-left: auto" class="card card-border bg-base-100 w-96">
        <div class="card-body">
            <h2 class="card-title">Filtrovat data</h2>
            <fieldset class="fieldset">
                <legend class="fieldset-legend">Povětrnostní podmínky</legend>
                <input type="text" class="input" placeholder="My awesome page" />
                <p class="fieldset-label">You can edit page title later on from settings</p>
            </fieldset>
            <fieldset class="fieldset">
                <legend class="fieldset-legend">Řidič</legend>
                <input type="text" class="input" placeholder="My awesome page" />
                <p class="fieldset-label">You can edit page title later on from settings</p>
            </fieldset>
            <div class="card-actions justify-end">
                <button class="btn btn-ghost">Zrušit filtr</button>
                <button class="btn btn-primary">Filtrovat</button>
            </div>
        </div>
    </div>
</template>

<style scoped>

</style>