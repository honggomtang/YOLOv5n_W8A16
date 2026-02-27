`timescale 1ns / 1ps

(* use_dsp = "yes" *)
module conv_acc_requant #(
    parameter NUM_CHANNELS = 64,
    parameter IN_WIDTH     = 32,
    parameter OUT_WIDTH    = 16
)(
    input  wire                     aclk, aresetn,
    input  wire                     in_valid,
    input  wire                     in_last,
    input  wire [IN_WIDTH-1:0]      in_data,
    input  wire [31:0]              multiplier,
    input  wire                     out_ready,
    output reg                      out_valid,
    output reg                      out_last,
    output reg [OUT_WIDTH-1:0]      out_data,
    output wire                     can_accept
);
    reg [6:0] ch_ptr;
    reg busy;
    reg output_hold_valid;
    reg [OUT_WIDTH-1:0] output_hold_data;
    reg output_hold_last;

    reg signed [63:0] product_reg;
    reg        product_valid;
    reg        product_last;

    reg signed [63:0] shifted_reg;
    reg        shift_valid;
    reg        shift_last;

    reg        or_upper_r, or_lower_r;
    reg        and_upper_r, and_lower_r;
    reg [15:0] prep_16_r;
    reg        sign_r;
    reg        stage2a1_valid_r, stage2a1_last_r;

    reg        ovf_flag_r, unf_flag_r;
    reg [15:0] prep_data_r;
    reg        stage2a_valid_r, stage2a_last_r;

    reg [15:0] clamp_prep_reg;
    reg        clamp_prep_valid;
    reg        clamp_prep_last;

    wire signed [31:0] current_acc = in_data;
    wire signed [63:0] product = current_acc * $signed({1'b0, multiplier});

    wire signed [63:0] rounded  = $signed(product_reg) + 64'sd32768;
    wire signed [63:0] shifted  = rounded >>> 16;

    wire overflow  = !sign_r && (or_upper_r | or_lower_r | (prep_16_r >= 16'h8000));
    wire underflow = sign_r && (~(and_upper_r & and_lower_r) || (prep_16_r < 16'h8000));
    wire [15:0] final_clamped = ovf_flag_r ? 16'h7FFF : (unf_flag_r ? 16'h8000 : prep_data_r);

    assign can_accept = (!busy || (ch_ptr < NUM_CHANNELS)) && !output_hold_valid && (!product_valid || out_ready);

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            ch_ptr <= 0;
            busy <= 0;
            out_valid <= 0;
            out_last <= 0;
            out_data <= 0;
            output_hold_valid <= 0;
            output_hold_data <= 0;
            output_hold_last <= 0;
            product_reg <= 0;
            product_valid <= 0;
            product_last <= 0;
            shifted_reg <= 0;
            shift_valid <= 0;
            shift_last <= 0;
            or_upper_r <= 0; or_lower_r <= 0; and_upper_r <= 0; and_lower_r <= 0;
            prep_16_r <= 0; sign_r <= 0; stage2a1_valid_r <= 0; stage2a1_last_r <= 0;
            ovf_flag_r <= 0; unf_flag_r <= 0; prep_data_r <= 0;
            stage2a_valid_r <= 0; stage2a_last_r <= 0;
            clamp_prep_reg <= 0;
            clamp_prep_valid <= 0;
            clamp_prep_last <= 0;
        end else begin
            if (output_hold_valid && out_ready)
                output_hold_valid <= 0;
            if (out_valid && out_ready)
                out_valid <= 0;

            if (in_valid && can_accept) begin
                if (!busy) busy <= 1'b1;
                product_reg   <= product;
                product_valid <= 1;
                product_last  <= (ch_ptr == NUM_CHANNELS - 1) ? in_last : 1'b0;
                if (ch_ptr == NUM_CHANNELS - 1) begin
                    busy <= 0;
                    ch_ptr <= 0;
                end else
                    ch_ptr <= ch_ptr + 1;
            end

            if (clamp_prep_valid) begin
                clamp_prep_valid <= 0;
                if (out_ready) begin
                    out_valid <= 1;
                    out_data  <= clamp_prep_reg;
                    out_last  <= clamp_prep_last;
                end else begin
                    output_hold_valid <= 1;
                    output_hold_data  <= clamp_prep_reg;
                    output_hold_last  <= clamp_prep_last;
                end
            end
            if (!clamp_prep_valid && !output_hold_valid) begin
                out_valid <= 0;
                out_last  <= 0;
            end

            if (stage2a_valid_r) begin
                stage2a_valid_r <= 0;
                clamp_prep_reg   <= final_clamped;
                clamp_prep_valid <= 1;
                clamp_prep_last  <= stage2a_last_r;
            end
            if (stage2a1_valid_r) begin
                stage2a1_valid_r <= 0;
                ovf_flag_r  <= overflow;
                unf_flag_r  <= underflow;
                prep_data_r <= prep_16_r;
                stage2a_valid_r <= 1;
                stage2a_last_r  <= stage2a1_last_r;
            end
            if (shift_valid) begin
                shift_valid <= 0;
                or_upper_r  <= |shifted_reg[62:40];
                or_lower_r  <= |shifted_reg[39:16];
                and_upper_r <= &shifted_reg[62:40];
                and_lower_r <= &shifted_reg[39:16];
                prep_16_r   <= shifted_reg[15:0];
                sign_r      <= shifted_reg[63];
                stage2a1_valid_r <= 1;
                stage2a1_last_r  <= shift_last;
            end

            if (product_valid) begin
                if (!(in_valid && can_accept)) product_valid <= 0;
                shifted_reg <= shifted;
                shift_valid <= 1;
                shift_last  <= product_last;
            end else begin
                shift_valid <= 0;
            end

            if (output_hold_valid && out_ready) begin
                out_valid <= 1;
                out_data  <= output_hold_data;
                out_last  <= output_hold_last;
            end
        end
    end
endmodule
