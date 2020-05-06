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
            //$('#result').attr('src', 'data:image/png;base64, iVBORw0KGgoAAAANSUhEUgAAAAUAAAAFCAYAAACNbyblAAAAHElEQVQI12P4//8/w38GIAXDIBKE0DHxgljNBAAO9TXL0Y4OHwAAAABJRU5ErkJggg==');
            //$('#Para').text('NEW')
        });
        return false;
    });
});

