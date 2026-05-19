window.onload = function () {
    const ctx = document.getElementById("myChart");

    if (ctx) {
        new Chart(ctx, {
            type: "bar",
            data: {
                labels: ["Pasien"],
                datasets: [{
                    label: "Jumlah Data",
                    data: [window.totalPasien],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        });
    }
};