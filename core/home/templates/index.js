$('btn btn-primary btn-lg px-4 gap-3').addEventListener('click', function() {
    $.ajax({
        url: '/detect',
        method: 'GET',
        success: function(data) {
            console.log(data);
        }
    });
});
