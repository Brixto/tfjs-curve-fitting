var xs = []
var ys = []
var modelTrained;

document.getElementById('x').value = 1; 

document.getElementById("append").onclick = function(){
        var x = document.getElementById("x").value;
        var y = document.getElementById("y").value;

        xs.push(x)
        ys.push(y)

        document.getElementById('x').value = parseInt(x)+1;

        trainModel(xs, ys).then(h => {
                console.log(h);
                modelTrained = h.model;
                var bestfit = h.model.predict(tf.tensor(xs, [xs.length, 1])).dataSync();
                renderChart(xs, ys, bestfit);
        })
}

document.getElementById("predict").onclick = function() {
        var predv = document.getElementById("predictv").value;
        console.log(modelTrained.predict(tf.tensor([predv], [1,1])).dataSync());
}

function trainModel(xs, ys) {
        const model = tf.sequential();
        model.add(tf.layers.dense({units: 128, inputShape: [1]})); 
        model.add(tf.layers.dense({units: 128, inputShape: [128], activation: "sigmoid"})); 
        model.add(tf.layers.dense({units: 1, inputShape: [128]})); 

        var optimizer = tf.train.adam(0.1);

        model.compile({loss: 'meanSquaredError', optimizer: optimizer});

        return model.fit(tf.tensor(xs), tf.tensor(ys), {epochs:300})
}

function renderChart(xs, ys, bestfit) {
        var ctx = document.getElementById("myChart").getContext('2d');
        var myChart = new Chart(ctx, {
                type: 'line',
                options: {scales:{yAxes: [{ticks: {beginAtZero: true}}]}},
                data: {
                    labels: xs,
                    datasets: [
                    {
                        label: 'Original Data',
                        data: ys,
                        borderWidth: 1,
                    }, {
                        label: 'Best Fit',
                        data: bestfit,
                        borderWidth: 1,
                        borderColor: '#FF0000',
                        backgroundColor: 'rgba(1,1,1,0)'
                    }]
                },
            });
}