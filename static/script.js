/* ============================================================
   Client-side form handler and API interface
   ============================================================ */

document.addEventListener("DOMContentLoaded", () => {

    const form = document.getElementById("prediction-form");

    const submitBtn =
        document.getElementById("submit-btn");

    const resultDisplay =
        document.getElementById("result-display");

    const errorMessage =
        document.getElementById("error-message");


    // ========================================================
    // FORM SUBMISSION
    // ========================================================

    form.addEventListener("submit", async (event) => {

        event.preventDefault();


        // ----------------------------------------------------
        // Reset previous result/error
        // ----------------------------------------------------

        errorMessage.classList.add("hidden");

        errorMessage.textContent = "";

        submitBtn.disabled = true;

        submitBtn.textContent = "Calculating...";

        resultDisplay.innerHTML = `
            <span class="placeholder-text">
                Processing request...
            </span>
        `;


        // ====================================================
        // COLLECT FORM DATA
        // ====================================================

        const formData = new FormData(form);


        // ====================================================
        // CREATE JSON PAYLOAD
        // ====================================================

        const payload = {

            // Hour must be a number
            hour: parseInt(
                formData.get("hour"),
                10
            ),

            // Day remains a string
            day_of_week:
                formData.get("day_of_week"),

            // IMPORTANT:
            // Priority is now a STRING because the
            // dataset contains 1, 2, 3, E and I.
            priority:
                formData.get("priority"),

            // Unit type
            unit_type:
                formData.get("unit_type"),

            // Location
            zipcode:
                formData.get("zipcode"),

            station_area:
                formData.get("station_area"),

            neighborhood:
                formData.get("neighborhood"),

            // Checkbox
            als_unit:
                document.getElementById(
                    "als_unit"
                ).checked
        };


        // ====================================================
        // SEND REQUEST TO FLASK
        // ====================================================

        try {

            const response = await fetch(
                "/predict",
                {
                    method: "POST",

                    headers: {
                        "Content-Type":
                            "application/json"
                    },

                    body:
                        JSON.stringify(payload)
                }
            );


            // ------------------------------------------------
            // Read server response
            // ------------------------------------------------

            const data =
                await response.json();


            // ------------------------------------------------
            // Handle server error
            // ------------------------------------------------

            if (!response.ok) {

                throw new Error(
                    data.error ||
                    "An error occurred while generating prediction."
                );
            }


            // =================================================
            // DISPLAY SUCCESSFUL PREDICTION
            // =================================================

            resultDisplay.innerHTML = `

                <div class="result-value">
                    ${data.predicted_response_time_minutes}
                </div>

                <div class="result-unit">
                    minutes
                </div>

            `;


        } catch (error) {


            // =================================================
            // DISPLAY ERROR
            // =================================================

            resultDisplay.innerHTML = `

                <span class="placeholder-text">
                    Prediction unavailable
                </span>

            `;


            errorMessage.textContent =
                error.message;

            errorMessage.classList.remove(
                "hidden"
            );


        } finally {


            // =================================================
            // RESTORE BUTTON
            // =================================================

            submitBtn.disabled = false;

            submitBtn.textContent =
                "Predict Response Time";

        }

    });

});