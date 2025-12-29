function startLoading(statusText) {
    $("#loadingContainer").removeClass("d-none");
    updateLoading(0, statusText);

    let progress = 0;
    window.loadingInterval = setInterval(() => {
        // Increase progress up to ~90% while waiting
        if (progress < 90) {
            progress += Math.floor(Math.random() * 10) + 5; // random increments
            if (progress > 90) progress = 90;
            updateLoading(progress, statusText);
        }
    }, 400);
}

function updateLoading(percent, statusText) {
    $("#loadingBar").css("width", percent + "%").text(percent + "%");
    if (statusText) {
        $("#loadingStatus").text(statusText);
    }
}

function stopLoading() {
    clearInterval(window.loadingInterval);
    updateLoading(100, "Done!");
    setTimeout(() => {
        $("#loadingContainer").addClass("d-none");
    }, 500);
}

$(function(){
    $("#askBtn").click(function(){
        let question = $("#question").val().trim();
        $("#error").addClass("d-none");
        $("#sqlQuery").text("");
        $("#finalAnswer").text("");

        if ($.fn.DataTable.isDataTable("#resultsTable")) {
            $("#resultsTable").DataTable().clear().destroy();
        }
        $("#tableHead").empty();
        $("#tableBody").empty();

        if(!question) {
            $("#error").text("Please enter a question.").removeClass("d-none");
            return;
        }

        startLoading("Processing question...");

        const startTime = performance.now();

        $.ajax({
            url: "/ask",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({question: question}),
            success: function(res){
                stopLoading();

                const execMs = (performance.now() - startTime).toFixed(0);
                $("#execTime").text(execMs + " ms");

                if(res.error){
                    $("#error").text(res.error).removeClass("d-none");
                    return;
                }

                $("#sqlEditor").val(res.sql);
                $("#sqlQuery").text(res.sql);
                const htmlAnswer = marked.parse(res.answer);
                $("#finalAnswer").html(htmlAnswer);

                const dtColumns = res.columns.map(col => ({ title: col, data: col }));
                $("#resultsTable").DataTable({
                    data: res.data,
                    columns: dtColumns,
                    destroy: true,
                    paging: true,
                    searching: true,
                    info: true,
                    autoWidth: false
                });
            },
            error: function(){
                stopLoading();
                $("#error").text("An unexpected error occurred.").removeClass("d-none");
            }
        });
    });

    $("#runSqlBtn").click(function(){
        let sql = $("#sqlEditor").val().trim();
        if(!sql) {
            $("#error").text("Please enter SQL to run.").removeClass("d-none");
            return;
        }

        $("#error").addClass("d-none");
        if ($.fn.DataTable.isDataTable("#resultsTable")) {
            $("#resultsTable").DataTable().clear().destroy();
        }
        $("#tableHead").empty();
        $("#tableBody").empty();

        startLoading("Running SQL query...");

        const startTime = performance.now();

        $.ajax({
            url: "/run_sql",
            type: "POST",
            contentType: "application/json",
            data: JSON.stringify({sql: sql}),
            success: function(res){
                stopLoading();

                const execMs = (performance.now() - startTime).toFixed(0);
                $("#execTime").text(execMs + " ms");

                if(res.error){
                    $("#error").text(res.error).removeClass("d-none");
                    return;
                }

                const dtColumns = res.columns.map(col => ({ title: col, data: col }));
                $("#resultsTable").DataTable({
                    data: res.data,
                    columns: dtColumns,
                    destroy: true,
                    paging: true,
                    searching: true,
                    info: true,
                    autoWidth: false
                });
            },
            error: function(){
                stopLoading();
                $("#error").text("An unexpected error occurred.").removeClass("d-none");
            }
        });
    });
});
