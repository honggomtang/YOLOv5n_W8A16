`timescale 1ns / 1ps

(* use_dsp = "yes" *)
(* use_dsp48 = "yes" *)
module pe_mac(
    input wire aclk, aresetn,
    input wire en,
    input wire clear_acc,
    input wire signed [15:0] din_a,
    input wire signed [7:0]  din_w,
    input wire signed [31:0] bias,
    output reg signed [31:0] dout
);
    reg signed [15:0] din_a_r;
    reg signed [7:0]  din_w_r;
    reg en_r, clear_acc_r;
    reg signed [31:0] bias_r;
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            din_a_r <= 0; din_w_r <= 0; en_r <= 0; clear_acc_r <= 0; bias_r <= 0;
        end else begin
            din_a_r <= din_a; din_w_r <= din_w; en_r <= en; clear_acc_r <= clear_acc; bias_r <= bias;
        end
    end
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn)
            dout <= 0;
        else if (en_r) begin
            if (clear_acc_r)
                dout <= bias_r + (din_a_r * din_w_r);
            else
                dout <= dout + (din_a_r * din_w_r);
        end
    end
endmodule
