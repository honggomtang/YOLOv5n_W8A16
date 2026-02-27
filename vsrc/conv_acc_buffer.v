`timescale 1ns / 1ps

module conv_acc_buffer #(
    parameter DATA_WIDTH       = 32,
    parameter W_ADDR_WIDTH    = 11,
    parameter B_ADDR_WIDTH   = 7,
    parameter L_ADDR_WIDTH   = 12,
    parameter NUM_WEIGHT_BANKS = 8,
    parameter NUM_LINES      = 6,
    parameter MAX_W          = 3072
)(
    input  wire                   aclk, aresetn,
    output wire                   s_axis_tready,
    input  wire [DATA_WIDTH-1:0]  s_axis_tdata,
    input  wire                   s_axis_tvalid, s_axis_tlast,
    input  wire [3:0]             kernel_size,
    input  wire                   start_load,
    input  wire                   row_done,
    input  wire [W_ADDR_WIDTH-1:0] rd_w_addr,
    input  wire [B_ADDR_WIDTH-1:0] rd_b_addr,
    input  wire [L_ADDR_WIDTH-1:0] rd_l_addr,
    output wire [2:0]             curr_line_idx,
    output reg                    is_1x1_path,
    output wire [NUM_WEIGHT_BANKS*32-1:0] weight_data_all,
    output wire [DATA_WIDTH-1:0]  bias_data,
    output wire [DATA_WIDTH-1:0]  act_data_line0, act_data_line1, act_data_line2,
    output wire [DATA_WIDTH-1:0]  act_data_line3, act_data_line4, act_data_line5,
    output reg                    buffer_ready,
    output wire                   in_load_phase
);
    localparam IDLE = 2'b00, LOAD_B = 2'b01, LOAD_W = 2'b10, STREAM_ACT = 2'b11;
    reg start_load_reg;
    reg rst_done;
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) rst_done <= 1'b0;
        else          rst_done <= 1'b1;
    end
    wire start_load_pos_edge = start_load && !start_load_reg && rst_done;

    reg [1:0] state;

    reg [B_ADDR_WIDTH-1:0] wr_b_ptr;
    reg [W_ADDR_WIDTH-1:0] wr_w_ptr;
    reg [3:0]              wr_w_sel;
    reg [L_ADDR_WIDTH-1:0] wr_l_ptr;
    reg [2:0] line_idx, filled_line_count;

    assign s_axis_tready = (state != IDLE);
    assign curr_line_idx = line_idx;
    assign in_load_phase = (state == LOAD_B || state == LOAD_W);

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) start_load_reg <= 1'b0;
        else          start_load_reg <= start_load;
    end

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            state <= IDLE; {wr_b_ptr, wr_w_ptr, wr_w_sel, wr_l_ptr} <= 0;
            line_idx <= 0; buffer_ready <= 0; filled_line_count <= 0;
        end else begin
            case (state)
                IDLE: if (start_load_pos_edge) begin state <= LOAD_B; is_1x1_path <= (kernel_size == 4'd1); end
                LOAD_B: if (s_axis_tvalid && s_axis_tready) begin
                    wr_b_ptr <= wr_b_ptr + 1;
                    if (s_axis_tlast) begin state <= LOAD_W; wr_b_ptr <= 0; wr_w_sel <= 0; wr_w_ptr <= 0; end
                end
                LOAD_W: if (s_axis_tvalid && s_axis_tready) begin
                    if (wr_w_sel == NUM_WEIGHT_BANKS - 1) begin wr_w_sel <= 0; wr_w_ptr <= wr_w_ptr + 1; end
                    else wr_w_sel <= wr_w_sel + 1;
                    if (s_axis_tlast) begin
                        state <= STREAM_ACT; wr_w_ptr <= 0; wr_w_sel <= 0;
                    end
                end
                STREAM_ACT: begin
                    if (start_load_pos_edge && !row_done) begin
                        state <= LOAD_B; is_1x1_path <= (kernel_size == 4'd1);
                        wr_l_ptr <= 0; line_idx <= 0; filled_line_count <= 0; buffer_ready <= 0;
                    end else if (row_done) begin
                        state <= STREAM_ACT;
                        buffer_ready <= 0;
                        wr_l_ptr <= 0;
                        line_idx <= 0;
                        filled_line_count <= 0;
                    end else begin
                        if (s_axis_tvalid && s_axis_tready) begin
                            if (s_axis_tlast || (wr_l_ptr == MAX_W - 1)) begin
                                wr_l_ptr <= 0;
                                line_idx <= (line_idx == NUM_LINES - 1) ? 0 : line_idx + 1;
                                if (filled_line_count < NUM_LINES) filled_line_count <= filled_line_count + 1;
                            end else wr_l_ptr <= wr_l_ptr + 1;
                        end
                        case (kernel_size)
                            4'd1: buffer_ready <= (filled_line_count >= 1);
                            4'd3: buffer_ready <= (filled_line_count >= 3);
                            4'd6: buffer_ready <= (filled_line_count >= 6);
                            default: buffer_ready <= (filled_line_count >= NUM_LINES);
                        endcase
                    end
                end
            endcase
        end
    end

    (* ram_style = "block" *) reg [31:0] bias_buf [0:(2**B_ADDR_WIDTH)-1];
    reg [31:0] bias_data_r;
    always @(posedge aclk)
        if (state == LOAD_B && s_axis_tvalid && s_axis_tready)
            bias_buf[wr_b_ptr] <= s_axis_tdata;
    always @(posedge aclk) bias_data_r <= bias_buf[rd_b_addr];
    assign bias_data = bias_data_r;

    genvar i;
    generate
        for (i = 0; i < NUM_WEIGHT_BANKS; i = i + 1) begin : gen_weight_mems
            (* ram_style = "block" *) reg [31:0] weight_bank [0:(2**W_ADDR_WIDTH)-1];
            reg [31:0] weight_data_r;
            always @(posedge aclk) begin
                if (state == LOAD_W && s_axis_tvalid && wr_w_sel == i)
                    weight_bank[wr_w_ptr] <= s_axis_tdata;
            end
            always @(posedge aclk) weight_data_r <= weight_bank[rd_w_addr];
            assign weight_data_all[i*32 +: 32] = weight_data_r;
        end
    endgenerate

    (* ram_style = "block" *) reg [31:0] line_buf_0 [0:MAX_W-1];
    (* ram_style = "block" *) reg [31:0] line_buf_1 [0:MAX_W-1];
    (* ram_style = "block" *) reg [31:0] line_buf_2 [0:MAX_W-1];
    (* ram_style = "block" *) reg [31:0] line_buf_3 [0:MAX_W-1];
    (* ram_style = "block" *) reg [31:0] line_buf_4 [0:MAX_W-1];
    (* ram_style = "block" *) reg [31:0] line_buf_5 [0:MAX_W-1];

    always @(posedge aclk)
        if (state == STREAM_ACT && s_axis_tvalid && s_axis_tready && (line_idx == 3'd0))
            line_buf_0[wr_l_ptr] <= s_axis_tdata;
    always @(posedge aclk)
        if (state == STREAM_ACT && s_axis_tvalid && s_axis_tready && !is_1x1_path && (line_idx == 3'd1))
            line_buf_1[wr_l_ptr] <= s_axis_tdata;
    always @(posedge aclk)
        if (state == STREAM_ACT && s_axis_tvalid && s_axis_tready && !is_1x1_path && (line_idx == 3'd2))
            line_buf_2[wr_l_ptr] <= s_axis_tdata;
    always @(posedge aclk)
        if (state == STREAM_ACT && s_axis_tvalid && s_axis_tready && !is_1x1_path && (line_idx == 3'd3))
            line_buf_3[wr_l_ptr] <= s_axis_tdata;
    always @(posedge aclk)
        if (state == STREAM_ACT && s_axis_tvalid && s_axis_tready && !is_1x1_path && (line_idx == 3'd4))
            line_buf_4[wr_l_ptr] <= s_axis_tdata;
    always @(posedge aclk)
        if (state == STREAM_ACT && s_axis_tvalid && s_axis_tready && !is_1x1_path && (line_idx == 3'd5))
            line_buf_5[wr_l_ptr] <= s_axis_tdata;

    reg [31:0] act_r0, act_r1, act_r2, act_r3, act_r4, act_r5;
    always @(posedge aclk) act_r0 <= line_buf_0[rd_l_addr];
    always @(posedge aclk) act_r1 <= line_buf_1[rd_l_addr];
    always @(posedge aclk) act_r2 <= line_buf_2[rd_l_addr];
    always @(posedge aclk) act_r3 <= line_buf_3[rd_l_addr];
    always @(posedge aclk) act_r4 <= line_buf_4[rd_l_addr];
    always @(posedge aclk) act_r5 <= line_buf_5[rd_l_addr];
    assign act_data_line0 = act_r0;
    assign act_data_line1 = act_r1;
    assign act_data_line2 = act_r2;
    assign act_data_line3 = act_r3;
    assign act_data_line4 = act_r4;
    assign act_data_line5 = act_r5;
endmodule
