`timescale 1ns / 1ps

module pe_cluster #(
    parameter USE_DSP = 1
)(
    input  wire         aclk, aresetn,
    input  wire         en,
    input  wire         clear_acc,
    input  wire         is_padding,
    input  wire [15:0]  din_a,
    input  wire [31:0]  din_w_4pack,
    input  wire [127:0] bias_4pack,
    output wire [127:0] dout_4pack
);
    wire [15:0] safe_din_a;
    assign safe_din_a = (is_padding) ? 16'd0 : din_a;

    genvar i;
    generate
        for (i = 0; i < 4; i = i + 1) begin : gen_pe
            pe_mac u_pe (
                .aclk(aclk), .aresetn(aresetn), .en(en),
                .clear_acc(clear_acc),
                .din_a(safe_din_a),
                .din_w(din_w_4pack[i*8 +: 8]),
                .bias(bias_4pack[i*32 +: 32]),
                .dout(dout_4pack[i*32 +: 32])
            );
        end
    endgenerate
endmodule
