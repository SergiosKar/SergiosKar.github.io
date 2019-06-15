
/*-------------------------------------------------------------------------*/
/* FORM VALIDATION */
/* -----------------------------------------------------------------------*/

function formCheck() {
    $( ".js-submit" ).click( function( e ) {

        e.preventDefault();

        var $inputs = $( ".form__input input" );
        var textarea = $( ".form__input textarea" );
        var isError = false;

        $( ".form__input" ).removeClass( "error" );
        $( ".error-data" ).remove();

        for ( var i = 0; i < $inputs.length; i++ ) {
            var input = $inputs[ i ];
            if ( $( input ).attr( "required", true ) && !validateRequired( $( input ).val() ) ) {

                addErrorData( $( input ), "This field is required" );

                isError = true;
            }
            if ( $( input ).attr( "required", true ) && $( input ).attr( "type" ) === "email" && !validateEmail( $( input ).val() ) ) {
                addErrorData( $( input ), "Email address is invalid" );
                isError = true;
            }
            // if ( $( textarea ).attr( "required", true ) && !validateRequired( $( textarea ).val() ) ) {
            //     addErrorData( $( textarea ), "This field is required" );
            //     isError = true;
            // }
        }
        if ( isError === false ) {
            $( "#contactForm" ).submit();
        }
    } );
}

// Validate if the input is not empty
function validateRequired( value ) {
    if ( value === "" ) {
      return false;
    }
    return true;
}

// Validate if the email is using correct format
function validateEmail( value ) {
    if ( value !== "" ) {
        return /[a-z0-9!#$%&'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&'*+/=?^_`{|}~-]+)*@(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?/i.test( value );
    }
    return true;
}

// Add error message to the input
function addErrorData( element, error ) {
    element.parent().addClass( "error" );
    element.after( "<span class='error-data'>" + error + "</span>" );
}

