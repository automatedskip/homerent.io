$(function () {
    $('#calculate').bind('click', function () {
        $.getJSON($SCRIPT_ROOT + '/_add_plots', {
            city: $('input[name="city"]').val(),
            state: $('input[name="state"]').val(),
            full_state: $('input[name="full_state"]').val()
        }, function (data) {
            //$('#result').attr('src', 'data:image/png;base64,' + data.result);
            $('#result').attr('src', 'data:image/png;base64,'.concat(data.result));
            $('#result2').attr('src', 'data:image/png;base64,'.concat(data.result2));
            $('#result3').attr('src', 'data:image/png;base64,'.concat(data.result3));
            $('#result4').attr('src', 'data:image/png;base64,'.concat(data.result4));
            $('#result6').attr('src', 'data:image/png;base64,'.concat(data.result6));
            $('#result7').attr('src', 'data:image/png;base64,'.concat(data.result7));
            $("#result5").text(data.result5);
            $("#result8").text(data.result8);
            $("#result9").text(data.result9);
            $("#result10").text(data.result10);
        });
        return false;
    });
});

