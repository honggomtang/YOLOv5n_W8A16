`timescale 1ns / 1ps

module conv_acc_top #(
    parameter DATA_WIDTH    = 32,
    parameter W_ADDR_WIDTH  = 11,
    parameter B_ADDR_WIDTH  = 7,
    parameter L_ADDR_WIDTH  = 12,
    parameter NUM_CLUSTERS  = 8,
    parameter NUM_PE        = 32,
    parameter NUM_LINES     = 6,
    parameter MAX_W         = 3072
)(
    input  wire                   aclk, aresetn,

    output wire                   s_axis_tready,
    input  wire [DATA_WIDTH-1:0] s_axis_tdata,
    input  wire                   s_axis_tvalid, s_axis_tlast,

    input  wire [3:0]             kernel_size,
    input  wire [31:0]            target_ic,
    input  wire [15:0]            img_width,
    input  wire [31:0]            multiplier,
    input  wire [3:0]             stride_val,
    input  wire [6:0]             act_start,
    input  wire                   start_load,

    output wire                   m_axis_tvalid,
    output wire [31:0]            m_axis_tdata,
    output wire                   m_axis_tlast,
    input  wire                   m_axis_tready
);

    reg [3:0]  kernel_size_r;
    reg [31:0] target_ic_r;
    reg [15:0] img_width_r;
    reg [31:0] multiplier_r;
    reg [3:0]  stride_val_r;
    reg [6:0]  act_start_r;
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            kernel_size_r <= 0;
            target_ic_r   <= 0;
            img_width_r   <= 0;
            multiplier_r  <= 0;
            stride_val_r  <= 0;
            act_start_r   <= 0;
        end else begin
            kernel_size_r <= kernel_size;
            target_ic_r   <= target_ic;
            img_width_r   <= img_width;
            multiplier_r  <= multiplier;
            stride_val_r  <= stride_val;
            act_start_r   <= act_start;
        end
    end

    wire [W_ADDR_WIDTH-1:0] rd_w_addr;
    wire [B_ADDR_WIDTH-1:0] rd_b_addr;
    wire [L_ADDR_WIDTH-1:0] rd_l_addr;
    wire [2:0]              curr_line_idx;
    wire [NUM_CLUSTERS*32-1:0] weight_data_all;
    wire [DATA_WIDTH-1:0]   bias_data;
    wire [DATA_WIDTH-1:0]   act_lines [0:5];
    wire                    buffer_ready;
    wire                    is_1x1_path;
    wire                    row_done;
    wire                    in_load_phase;

    wire                    compute_valid;
    wire                    compute_last;
    wire                    engine_busy;
    wire [NUM_PE*32-1:0]    compute_data;
    wire                    ser_valid;
    wire [31:0]             ser_data;
    wire                    ser_last;
    wire                    ser_can_accept;
    wire                    requant_valid;
    wire                    requant_last;
    wire [15:0]             requant_data_16;
    wire                    requant_can_accept;
    wire                    requant_out_ready;

    conv_acc_buffer #(
        .DATA_WIDTH(DATA_WIDTH),
        .W_ADDR_WIDTH(W_ADDR_WIDTH),
        .B_ADDR_WIDTH(B_ADDR_WIDTH),
        .L_ADDR_WIDTH(L_ADDR_WIDTH),
        .NUM_WEIGHT_BANKS(NUM_CLUSTERS),
        .NUM_LINES(NUM_LINES),
        .MAX_W(MAX_W)
    ) u_buffer (
        .aclk(aclk), .aresetn(aresetn),
        .s_axis_tready(s_axis_tready), .s_axis_tdata(s_axis_tdata),
        .s_axis_tvalid(s_axis_tvalid), .s_axis_tlast(s_axis_tlast),
        .kernel_size(kernel_size_r), .start_load(start_load),
        .row_done(row_done),
        .rd_w_addr(rd_w_addr), .rd_b_addr(rd_b_addr), .rd_l_addr(rd_l_addr),
        .curr_line_idx(curr_line_idx), .is_1x1_path(is_1x1_path),
        .weight_data_all(weight_data_all),
        .bias_data(bias_data),
        .act_data_line0(act_lines[0]), .act_data_line1(act_lines[1]), .act_data_line2(act_lines[2]),
        .act_data_line3(act_lines[3]), .act_data_line4(act_lines[4]), .act_data_line5(act_lines[5]),
        .buffer_ready(buffer_ready), .in_load_phase(in_load_phase)
    );

    conv_acc_compute #(
        .DATA_WIDTH(DATA_WIDTH),
        .W_ADDR_WIDTH(W_ADDR_WIDTH),
        .B_ADDR_WIDTH(B_ADDR_WIDTH),
        .L_ADDR_WIDTH(L_ADDR_WIDTH),
        .NUM_CLUSTERS(NUM_CLUSTERS),
        .NUM_LINES(NUM_LINES)
    ) u_compute (
        .aclk(aclk), .aresetn(aresetn),
        .kernel_size(kernel_size_r), .buffer_ready(buffer_ready), .in_load_phase(in_load_phase),
        .target_ic(target_ic_r), .img_width(img_width_r), .stride_val(stride_val_r), .act_start(act_start_r),
        .rd_w_addr(rd_w_addr), .rd_b_addr(rd_b_addr), .rd_l_addr(rd_l_addr),
        .curr_line_idx(curr_line_idx),
        .weight_data_all(weight_data_all),
        .bias_data(bias_data),
        .act_data_line0(act_lines[0]), .act_data_line1(act_lines[1]), .act_data_line2(act_lines[2]),
        .act_data_line3(act_lines[3]), .act_data_line4(act_lines[4]), .act_data_line5(act_lines[5]),
        .out_ready(ser_can_accept),
        .engine_busy(engine_busy), .row_done(row_done),
        .out_valid(compute_valid), .out_last(compute_last),
        .out_data_pe(compute_data)
    );

    reg [31:0] ser_stages [0:1] [0:NUM_PE-1];
    reg [6:0]  ser_cnt;
    reg        ser_fill_slot;
    reg        ser_drain_slot;
    reg [1:0]  ser_num_frames;
    reg        ser_last_r [0:1];
    reg [31:0] ser_data_q;
    reg        ser_valid_q, ser_last_q;
    integer si;
    wire ser_has_data = (ser_num_frames != 0);
    wire ser_has_room = (ser_num_frames < 2);
    wire do_fill      = compute_valid && ser_has_room;
    wire do_drain     = ser_has_data && requant_can_accept;
    wire do_drain_last = do_drain && (ser_cnt == NUM_PE - 1);
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            ser_cnt <= 0;
            ser_fill_slot <= 0;
            ser_drain_slot <= 0;
            ser_num_frames <= 0;
            ser_last_r[0] <= 0; ser_last_r[1] <= 0;
            ser_data_q <= 0;
            ser_valid_q <= 0;
            ser_last_q <= 0;
        end else begin
            if (ser_last_q && requant_can_accept) begin
                ser_valid_q <= 0;
                ser_last_q  <= 0;
            end else if (!ser_has_data && !ser_last_q)
                { ser_valid_q, ser_last_q } <= 0;
            if (do_fill) begin
                for (si = 0; si < NUM_PE; si = si + 1)
                    ser_stages[ser_fill_slot][si] <= compute_data[si*32 +: 32];
                ser_fill_slot <= ~ser_fill_slot;
                ser_last_r[ser_fill_slot] <= compute_last;
            end
            if (do_drain) begin
                ser_data_q <= ser_stages[ser_drain_slot][ser_cnt];
                ser_valid_q <= 1;
                ser_last_q <= (ser_cnt == NUM_PE - 1) ? ser_last_r[ser_drain_slot] : 1'b0;
                if (ser_cnt == NUM_PE - 1) begin
                    ser_cnt <= 0;
                    ser_drain_slot <= ~ser_drain_slot;
                end else
                    ser_cnt <= ser_cnt + 1;
            end
            ser_num_frames <= ser_num_frames + (do_fill ? 1'd1 : 1'd0) - (do_drain_last ? 1'd1 : 1'd0);
        end
    end
    assign ser_can_accept = ser_has_room;
    assign ser_valid = ser_valid_q;
    assign ser_data = ser_data_q;
    assign ser_last = ser_last_q;

    conv_acc_requant #(.NUM_CHANNELS(NUM_PE), .IN_WIDTH(32), .OUT_WIDTH(16)) u_requant (
        .aclk(aclk), .aresetn(aresetn),
        .in_valid(ser_valid), .in_last(ser_last), .in_data(ser_data),
        .multiplier(multiplier_r),
        .out_ready(requant_out_ready),
        .out_valid(requant_valid), .out_last(requant_last), .out_data(requant_data_16),
        .can_accept(requant_can_accept)
    );

    localparam SER_WORDS = NUM_PE * 16 / 32;

    reg [31:0]  pack_reg;
    reg         pack_has_half;
    reg         pack_has_full;
    reg [4:0]   send_cnt;
    reg         last_word_flag;
    reg         next_word_is_last;

    assign requant_out_ready = !pack_has_full || m_axis_tready;

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            pack_reg <= 0; pack_has_half <= 0; pack_has_full <= 0;
            send_cnt <= 0;
            last_word_flag <= 0;
            next_word_is_last <= 0;
        end else begin
            if (pack_has_full && m_axis_tready) begin
                pack_has_full <= 0;
                last_word_flag <= 0;
                if (send_cnt == SER_WORDS - 1)
                    send_cnt <= 0;
                else
                    send_cnt <= send_cnt + 1;
            end
            if (requant_valid && requant_out_ready) begin
                if (pack_has_half) begin
                    pack_reg[31:16] <= requant_data_16;
                    pack_has_full <= 1;
                    pack_has_half <= 0;
                    last_word_flag <= next_word_is_last || requant_last;
                    if (requant_last)
                        next_word_is_last <= 0;
                end else begin
                    pack_reg[15:0] <= requant_data_16;
                    pack_has_half <= 1;
                    if (requant_last)
                        next_word_is_last <= 1;
                end
            end
        end
    end
    assign m_axis_tvalid = pack_has_full;
    assign m_axis_tdata  = pack_has_full ? pack_reg : 32'd0;
    assign m_axis_tlast  = pack_has_full && last_word_flag;
endmodule
