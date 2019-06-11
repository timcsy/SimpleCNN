<template>
<div>
    <canvas id="paint"></canvas>
    <button type="button" @click="sendjson">send</button>
</div>
</template>

<script>
    export default {
        name: "App",
        data() {
            return {
                //text: "Hello, Vue!"
                canvas : undefined
            };
        },
        methods: {
            test : function(event) {
                console.log(this.canvas.getContext()
                .getImageData(0,0,this.canvas.getWidth(),this.canvas.getHeight()))
            },
            sendjson : function(event) {
                console.log(JSON.stringify(this.canvas))
                console.log(this.canvas.toJSON())
                
                $.ajax({
                    type: 'POST',
                    url: '/json',
                    data: JSON.stringify(this.canvas),
                    success: function(data) { console.log('data: ' + data.msg); },
                    contentType: "application/json",
                    dataType: 'json'
                });
                
            }
        },
        mounted: function() {
            this.canvas = new fabric.Canvas('paint', {
                isDrawingMode : true,
                width: 560,
                height: 560
            });
            this.canvas.freeDrawingBrush.width = 20;
            this.canvas.freeDrawingBrush.color = 'red';
        }
    };
</script>

<style scoped>
    
</style>