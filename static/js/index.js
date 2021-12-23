$(document).ready(function () {
    $('#corp_info1').hide();
    $('#corp_info2').hide();
    $('#corp_info3').hide();
});

function onDisplay(id) {
    for (var i=1;i < 4; i++) {
        if (i != id) 
            $('#corp_info' + i).hide();
        else
            $('#corp_info' + id).show();       
    }
}