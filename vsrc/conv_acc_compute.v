`timescale 1ns / 1ps

module conv_acc_compute #(
    parameter DATA_WIDTH    = 32,
    parameter W_ADDR_WIDTH  = 11,
    parameter B_ADDR_WIDTH  = 7,
    parameter L_ADDR_WIDTH  = 12,
    parameter NUM_CLUSTERS  = 8,
    parameter NUM_LINES     = 6
)(
    input  wire                   aclk, aresetn,
    input  wire [3:0]             kernel_size,
    input  wire                   buffer_ready,
    input  wire                   in_load_phase,
    input  wire [31:0]            target_ic,
    input  wire [15:0]            img_width,
    input  wire [3:0]             stride_val,
    input  wire [6:0]             act_start,
    output reg [W_ADDR_WIDTH-1:0] rd_w_addr,
    output wire [B_ADDR_WIDTH-1:0] rd_b_addr,
    output reg [L_ADDR_WIDTH-1:0] rd_l_addr,
    input  wire [2:0]             curr_line_idx,
    input  wire [NUM_CLUSTERS*32-1:0] weight_data_all,
    input  wire [DATA_WIDTH-1:0]  bias_data,
    input  wire [DATA_WIDTH-1:0]  act_data_line0, act_data_line1, act_data_line2,
    input  wire [DATA_WIDTH-1:0]  act_data_line3, act_data_line4, act_data_line5,
    input  wire                   out_ready,
    output wire                   engine_busy,
    output wire                   row_done,
    output reg                    out_valid,
    output reg                    out_last,
    output reg [NUM_CLUSTERS*4*32-1:0] out_data_pe
);
    wire [NUM_CLUSTERS*4*32-1:0] pe_dout_raw;
    reg  [NUM_CLUSTERS*4*32-1:0] out_data_pe_r;
    localparam NUM_PE = NUM_CLUSTERS * 4;
    localparam IDLE = 3'd0, LOAD_BIAS = 3'd1, COMPUTE = 3'd2, DONE_WAIT = 3'd3, WAIT_NEXT_ROW = 3'd4;
    reg [2:0] state;

    reg [3:0]  kh_cnt, kw_cnt;
    reg [15:0] ic_cnt, col_cnt;
    reg [31:0] bias_regs [0:NUM_PE-1];
    reg [6:0]  bias_load_ptr;

    reg [L_ADDR_WIDTH-1:0] stride;
    reg [L_ADDR_WIDTH-1:0] pixel_start_addr, kw_base_addr;
    reg [B_ADDR_WIDTH-1:0] rd_b_addr_r;
    reg [3:0]              stride_val_r;

    assign rd_b_addr = (state == LOAD_BIAS) ? bias_load_ptr[B_ADDR_WIDTH-1:0] : rd_b_addr_r;

    wire [15:0] last_col = (stride_val_r == 4'd2) ?
        ((img_width - {12'b0, kernel_size}) >> 1) : (img_width - {12'b0, kernel_size});

    reg [8:0] valid_delay_pipe;
    reg compute_done_trigger;
    reg [8:0] last_pixel_delay_pipe;
    reg       last_pixel_trigger;
    reg [2:0] output_delay_cnt;
    reg [23:0] mac_count;

    assign engine_busy = (state != IDLE);
    assign row_done    = (state == DONE_WAIT);
    wire stalled = (kw_cnt == kernel_size - 1 && kh_cnt == kernel_size - 1 && ic_cnt == target_ic - 1 && !out_ready);
    wire in_delay   = (kw_cnt == kernel_size - 1 && kh_cnt == kernel_size - 1 && ic_cnt == target_ic - 1 && out_ready && (output_delay_cnt > 0));
    wire [31:0] mac_per_pixel = kernel_size * kernel_size * target_ic[15:0];
    wire in_transition = (state == LOAD_BIAS && bias_load_ptr > NUM_PE);
    wire pe_en_d1   = ((state == COMPUTE) || in_transition || (valid_delay_pipe > 0)) && !stalled && (mac_count < mac_per_pixel);
    reg first_pos_was_0;
    wire pe_clear_d1 = (first_pos_was_0 && (state == COMPUTE || in_transition))
                    || in_transition
                    || (state == WAIT_NEXT_ROW && buffer_ready);
    reg pe_en_d2, pe_clear_d2;
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) { pe_en_d2, pe_clear_d2 } <= 0;
        else begin pe_en_d2 <= pe_en_d1; pe_clear_d2 <= pe_clear_d1; end
    end

    wire signed [4:0] raw_line_idx   = curr_line_idx - kernel_size + kh_cnt;
    wire signed [4:0] wrapped_idx     = raw_line_idx + NUM_LINES;
    reg [2:0] physical_sel_r;
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn)
            physical_sel_r <= 0;
        else begin
            if (kernel_size == 4'd1)
                physical_sel_r <= 3'd0;
            else
                physical_sel_r <= (raw_line_idx >= 0) ? raw_line_idx[2:0] : wrapped_idx[2:0];
        end
    end

    wire [DATA_WIDTH-1:0] current_line = (physical_sel_r == 0) ? act_data_line0 :
                                         (physical_sel_r == 1) ? act_data_line1 :
                                         (physical_sel_r == 2) ? act_data_line2 :
                                         (physical_sel_r == 3) ? act_data_line3 :
                                         (physical_sel_r == 4) ? act_data_line4 : act_data_line5;
    reg ic_cnt_0_d1;
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) ic_cnt_0_d1 <= 0;
        else          ic_cnt_0_d1 <= ic_cnt[0];
    end
    wire [15:0] muxed_pixel = ic_cnt_0_d1 ? current_line[31:16] : current_line[15:0];
    reg [15:0] muxed_pixel_d1, muxed_pixel_d2;
    reg [NUM_CLUSTERS*32-1:0] weight_data_all_d1, weight_data_all_d2;
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            muxed_pixel_d1 <= 0; muxed_pixel_d2 <= 0;
            weight_data_all_d1 <= 0; weight_data_all_d2 <= 0;
        end else begin
            muxed_pixel_d1 <= muxed_pixel; muxed_pixel_d2 <= muxed_pixel_d1;
            weight_data_all_d1 <= weight_data_all; weight_data_all_d2 <= weight_data_all_d1;
        end
    end
    reg first_cycle_of_compute;
    wire [15:0] pe_act_in = muxed_pixel_d2;
    wire pe_clear_final = pe_clear_d3;
    wire pe_en_final   = pe_en_d4;
    wire [NUM_CLUSTERS*32-1:0] weight_to_pe = weight_data_all_d2;

    reg pe_en_d3, pe_clear_d3;
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) { pe_en_d3, pe_clear_d3 } <= 0;
        else begin pe_en_d3 <= pe_en_d2; pe_clear_d3 <= pe_clear_d2; end
    end
    reg pe_en_d4, pe_clear_d4;
    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) { pe_en_d4, pe_clear_d4 } <= 0;
        else begin pe_en_d4 <= pe_en_d3; pe_clear_d4 <= pe_clear_d3; end
    end

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            state <= IDLE;
            {rd_w_addr, rd_b_addr_r, rd_l_addr} <= 0;
            {ic_cnt, kh_cnt, kw_cnt, col_cnt, bias_load_ptr} <= 0;
            out_valid <= 0; out_last <= 0;
            valid_delay_pipe <= 0; compute_done_trigger <= 0;
            last_pixel_delay_pipe <= 0; last_pixel_trigger <= 0;
            {stride, pixel_start_addr, kw_base_addr} <= 0;
            stride_val_r <= 0;
            output_delay_cnt <= 0;
            first_cycle_of_compute <= 0;
            first_pos_was_0 <= 0;
            mac_count <= 0;
        end else begin
            if (state == COMPUTE || in_transition)
                first_pos_was_0 <= (kh_cnt == 0 && kw_cnt == 0 && ic_cnt == 0);
            if ((state == COMPUTE || in_transition) && !stalled)
                mac_count <= (mac_count < mac_per_pixel) ? (mac_count + 1) : mac_count;
            if (state == COMPUTE && first_cycle_of_compute && (kw_cnt != 0 || kh_cnt != 0 || ic_cnt != 0))
                first_cycle_of_compute <= 0;
            compute_done_trigger <= 1'b0;
            last_pixel_trigger   <= 1'b0;
            valid_delay_pipe <= {valid_delay_pipe[7:0], compute_done_trigger};
            last_pixel_delay_pipe <= {last_pixel_delay_pipe[7:0], last_pixel_trigger};
            out_valid <= valid_delay_pipe[7];
            out_last  <= last_pixel_delay_pipe[7];

            case (state)
                IDLE: if (buffer_ready) begin
                    state <= LOAD_BIAS; rd_b_addr_r <= 0; bias_load_ptr <= 0;
                    stride <= (target_ic + 1) >> 1; stride_val_r <= stride_val;
                    {pixel_start_addr, kw_base_addr, rd_w_addr} <= 0;
                end
                LOAD_BIAS: begin
                    if (bias_load_ptr > 0) bias_regs[bias_load_ptr-1] <= bias_data;
                    if (bias_load_ptr == NUM_PE) begin
                        rd_l_addr <= 0;
                        rd_w_addr <= 0;
                    end
                    if (bias_load_ptr <= NUM_PE) begin
                        bias_load_ptr <= bias_load_ptr + 1;
                    end else begin
                        state <= COMPUTE;
                        first_cycle_of_compute <= 1;
                        first_pos_was_0 <= 1;
                        pixel_start_addr <= {5'b0, act_start};
                        rd_l_addr <= {5'b0, act_start};
                        kw_base_addr <= {5'b0, act_start};
                        {ic_cnt, kh_cnt, kw_cnt, col_cnt, rd_w_addr, rd_b_addr_r} <= 0;
                        output_delay_cnt <= 0;
                        mac_count <= 0;
                    end
                end
                COMPUTE: begin
                    if (kw_cnt == kernel_size - 1 && kh_cnt == kernel_size - 1 && ic_cnt == target_ic - 1 && out_ready)
                        rd_l_addr <= pixel_start_addr;
                    else if (kh_cnt == 0 && kw_cnt == 0 && ic_cnt == 0)
                        rd_l_addr <= pixel_start_addr;
                    else if (kw_cnt == kernel_size - 1)
                        rd_l_addr <= pixel_start_addr + (ic_cnt >> 1);
                    else
                        rd_l_addr <= kw_base_addr + (ic_cnt >> 1);
                    if (kw_cnt == kernel_size - 1 && kh_cnt == kernel_size - 1 && ic_cnt == target_ic - 1)
                        rd_w_addr <= 0;
                    else if (kh_cnt == 0 && kw_cnt == 0 && ic_cnt == 0)
                        rd_w_addr <= 0;
                    else
                        rd_w_addr <= kw_cnt + (kh_cnt * kernel_size) + (ic_cnt * kernel_size * kernel_size);

                    if (kw_cnt == kernel_size - 1) begin
                        kw_base_addr <= pixel_start_addr;
                        if (kh_cnt == kernel_size - 1) begin
                            if (ic_cnt == target_ic - 1) begin
                                if (out_ready) begin
                                    if (output_delay_cnt == 3'd7) begin
                                        kw_cnt <= 0; kh_cnt <= 0; ic_cnt <= 0;
                                        output_delay_cnt <= 0;
                                        if (col_cnt == last_col) begin
                                            state <= DONE_WAIT;
                                        end else begin
                                            first_cycle_of_compute <= 1;
                                            first_pos_was_0 <= 1;
                                            mac_count <= 0;
                                            col_cnt <= col_cnt + 1;
                                            pixel_start_addr <= (stride_val_r == 4'd2) ?
                                                (pixel_start_addr + (stride << 1)) : (pixel_start_addr + stride);
                                            kw_base_addr <= (stride_val_r == 4'd2) ?
                                                (pixel_start_addr + (stride << 1)) : (pixel_start_addr + stride);
                                        end
                                    end else begin
                                        compute_done_trigger <= (output_delay_cnt == 3'd0);
                                        if (output_delay_cnt == 3'd0) last_pixel_trigger <= (col_cnt == last_col);
                                        output_delay_cnt <= output_delay_cnt + 1;
                                    end
                                end else begin
                                    kw_cnt <= kernel_size - 1; kh_cnt <= kernel_size - 1;
                                end
                                end else begin
                                    kw_cnt <= 0; kh_cnt <= 0; ic_cnt <= ic_cnt + 1;
                                end
                            end else begin
                                kw_cnt <= 0; kh_cnt <= kh_cnt + 1;
                            end
                        end else begin
                            kw_cnt <= kw_cnt + 1; kw_base_addr <= kw_base_addr + stride;
                        end
                end
                DONE_WAIT: if (!buffer_ready) state <= WAIT_NEXT_ROW;
                WAIT_NEXT_ROW: begin
                    if (in_load_phase) state <= IDLE;
                    else if (buffer_ready) begin
                        state <= COMPUTE;
                        first_cycle_of_compute <= 1;
                        first_pos_was_0 <= 1;
                        pixel_start_addr <= {5'b0, act_start};
                        rd_l_addr <= {5'b0, act_start};
                        kw_base_addr <= {5'b0, act_start};
                        {ic_cnt, kh_cnt, kw_cnt, col_cnt, rd_w_addr, rd_b_addr_r} <= 0;
                        stride <= (target_ic + 1) >> 1;
                        stride_val_r <= stride_val;
                        output_delay_cnt <= 0;
                        mac_count <= 0;
                    end
                end
            endcase
        end
    end

    always @(posedge aclk or negedge aresetn) begin
        if (!aresetn) begin
            out_data_pe_r <= 0;
            out_data_pe   <= 0;
        end else begin
            out_data_pe_r <= pe_dout_raw;
            out_data_pe   <= out_data_pe_r;
        end
    end

    genvar i;
    generate
        for (i = 0; i < NUM_CLUSTERS; i = i + 1) begin : gen_clusters
            pe_cluster #(.USE_DSP(1)) u_cluster (
                .aclk(aclk), .aresetn(aresetn),                 .en(pe_en_final),
                .clear_acc(pe_clear_final), .is_padding(1'b0), .din_a(pe_act_in),
                .din_w_4pack(weight_to_pe[i*32 +: 32]),
                .bias_4pack({bias_regs[i*4+3], bias_regs[i*4+2], bias_regs[i*4+1], bias_regs[i*4]}),
                .dout_4pack(pe_dout_raw[i*128 +: 128])
            );
        end
    endgenerate
endmodule
