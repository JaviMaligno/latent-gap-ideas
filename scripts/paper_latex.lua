-- Pandoc Lua filter for the arXiv LaTeX build of docs/paper/draft.md (scripts/build_paper_latex.py).
--
-- The Markdown keeps its typed numbering ("## 5 Distances ...", "**Table 2b.**", "Appendix A.8"); this filter turns it
-- into navigable LaTeX without changing what the reader sees:
--   * headings become real \section/\subsection with labels (sec:5.3, app:A.8); the typed number is checked
--     against LaTeX's counter, and "Appendix A/B" headings are numbered A, B after \appendix;
--   * every "Section N(.M)", "Appendix A(.N)/B", "Table N/AN/2b", "Figure N", a bare "A.N", and the section lists in
--     the "Section(s)" columns of tables become \hyperref links; author-year citations link their year to the entry;
--   * tables become floats (or longtables) with the caption paragraph above them attached as \caption, explicit
--     column widths and non-breaking spaces inside numbers; figures become [tb] floats with labels;
--   * the reference list gets anchors, linked arXiv identifiers and a hanging indent;
--   * inline code becomes \texttt that may break only after "_" or "/".

local stringify = pandoc.utils.stringify
local function raw(s) return pandoc.RawInline('latex', s) end
local function rawb(s) return pandoc.RawBlock('latex', s) end

local defined, used = {}, {}
local refkeys, refsurnames = {}, {}

---------------------------------------------------------------------------------------------------------- code spans
local function code_tex(text)
  local out = {}
  for _, c in utf8.codes(text) do
    local ch = utf8.char(c)
    if ch == '_' then out[#out + 1] = '\\_\\allowbreak{}'
    elseif ch == '/' then out[#out + 1] = '/\\allowbreak{}'
    elseif ch == '\\' then out[#out + 1] = '\\textbackslash{}'
    elseif ch:match('[{}$&#%%]') then out[#out + 1] = '\\' .. ch
    elseif ch == '^' then out[#out + 1] = '\\textasciicircum{}'
    elseif ch == '~' then out[#out + 1] = '\\textasciitilde{}'
    else out[#out + 1] = ch end
  end
  return '\\texttt{' .. table.concat(out) .. '}'
end

local function Code(el) return raw(code_tex(el.text)) end

------------------------------------------------------------------------------------------------------ cross-references
local KIND = { Section = 'sec', Sections = 'sec', Appendix = 'app', Table = 'tab', Tables = 'tab', Figure = 'fig',
               Figures = 'fig' }
local PATTERNS = {
  sec = { '^(%d+%.%d+)(.*)$', '^(%d+)(.*)$' },
  app = { '^([AC]%.%d)(.*)$', '^([ABC])(.*)$' },
  tab = { '^(A%d+)(.*)$', '^(%d+b?)(.*)$' },
  fig = { '^(%d+)(.*)$' },
  bare = { '^([AC]%.%d)(.*)$' },                      -- "A.8" without "Appendix"
  cell = { '^(%d+%.%d+)(.*)$', '^(%d+)(.*)$', '^([AC]%.%d)(.*)$' },  -- "Section(s)" table columns
}

local function label_for(kind, id)
  if kind == 'tab' then return 'tab:' .. id end
  if kind == 'fig' then return 'fig:' .. id end
  if id:match('^%d') then return 'sec:' .. id end
  return 'app:' .. id
end

local function match_id(kind, s)
  for _, pat in ipairs(PATTERNS[kind]) do
    local id, rest = s:match(pat)
    if id and not rest:match('^[A-Za-z0-9]') then return id, rest end
  end
end

local function link(label, inlines)
  used[label] = true
  local out = pandoc.Inlines { raw('\\hyperref[' .. label .. ']{') }
  out:extend(inlines)
  out:insert(raw('}'))
  return out
end

-- emit an id (and an en-dash range "A.1–A.4") with its trailing punctuation
local function emit_id(out, kind, id, rest, word)
  local inl = pandoc.Inlines {}
  if word then inl:insert(pandoc.Str(word)); inl:insert(raw('~')) end
  inl:insert(pandoc.Str(id))
  out:extend(link(label_for(kind == 'bare' and 'app' or kind, id), inl))
  local after = rest:match('^–(.*)$')
  if after then
    local id2, rest2 = match_id(kind == 'cell' and 'cell' or (kind == 'bare' and 'bare' or kind), after)
    if id2 then
      out:insert(pandoc.Str('–'))
      out:extend(link(label_for(kind == 'bare' and 'app' or kind, id2), pandoc.Inlines { pandoc.Str(id2) }))
      rest = rest2
    end
  end
  if rest ~= '' then out:insert(pandoc.Str(rest)) end
end

local function norm_name(t)
  t = t:gsub('’s$', ''):gsub("'s$", '')
  t = t:gsub('^%p+', ''):gsub('%p+$', '')
  t = t:gsub('’s$', ''):gsub("'s$", '')
  return t
end

local function anchor(key) return 'ref-' .. key:gsub('[^%w]', '') end

local function cite_key(inls, i, year)
  local steps = 0
  local j = i - 1
  while j >= 1 and steps < 12 do
    local el = inls[j]
    if el.t == 'Str' then
      steps = steps + 1
      local n = norm_name(el.text)
      if refkeys[n .. year] then return n .. year end
      if n:match('%d') then return nil end
      if not (n == 'et' or n == 'al' or n == 'and' or n == 'de' or n:match('^%u') or refsurnames[n]) then
        return nil
      end
    elseif el.t ~= 'Space' and el.t ~= 'SoftBreak' then
      return nil
    end
    j = j - 1
  end
end

local function link_inlines(inls)
  local out = pandoc.Inlines {}
  local i = 1
  local n = #inls
  while i <= n do
    local el = inls[i]
    local handled = false
    if el.t == 'Str' then
      local pre, word = el.text:match('^(%p*)(%a+)$')
      local kind = word and KIND[word]
      if kind and i + 2 <= n and (inls[i + 1].t == 'Space') and inls[i + 2].t == 'Str' then
        local id, rest = match_id(kind, inls[i + 2].text)
        if id then
          if pre ~= '' then out:insert(pandoc.Str(pre)) end
          emit_id(out, kind, id, rest, word)
          i = i + 3
          -- "Sections 3 to 5", "Tables 2 and 3"
          if rest == '' and i + 2 <= n and inls[i].t == 'Space' and inls[i + 1].t == 'Str'
              and (inls[i + 1].text == 'to' or inls[i + 1].text == 'and') and inls[i + 2].t == 'Space'
              and inls[i + 3] and inls[i + 3].t == 'Str' then
            local id2, rest2 = match_id(kind, inls[i + 3].text)
            if id2 then
              out:extend({ inls[i], inls[i + 1], inls[i + 2] })
              emit_id(out, kind, id2, rest2, nil)
              i = i + 4
            end
          end
          handled = true
        end
      end
      if not handled then
        local p0, rest0 = el.text:match('^(%p?)(A%.%d.*)$')
        if rest0 and not (i >= 3 and inls[i - 2].t == 'Str' and inls[i - 2].text:match('Appendix$')) then
          local id, rest = match_id('bare', rest0)
          if id then
            if p0 ~= '' then out:insert(pandoc.Str(p0)) end
            emit_id(out, 'bare', id, rest, nil)
            i = i + 1
            handled = true
          end
        end
      end
      if not handled then
        local pre, year, post = el.text:match('^(.-)(%d%d%d%d%l?)(%p*)$')
        -- pandoc glues "et al. (2026a)" into one Str with a non-breaking space
        if year and not (pre:match('^%p*$') or pre:match('^al%.\u{00A0}%(?$')) then year = nil end
        if year then
          local key = cite_key(inls, i, year)
          if key then
            if pre ~= '' then out:insert(pandoc.Str(pre)) end
            out:insert(raw('\\hyperlink{' .. anchor(key) .. '}{'))
            out:insert(pandoc.Str(year))
            out:insert(raw('}'))
            if post ~= '' then out:insert(pandoc.Str(post)) end
            if os.getenv('PAPER_DEBUG_CITES') then
              io.stderr:write('cite ' .. key .. ' <- ' .. stringify({ table.unpack(inls, math.max(1, i - 8), i) }) .. '\n')
            end
            i = i + 1
            handled = true
          end
        end
      end
    end
    if not handled then
      out:insert(el)
      i = i + 1
    end
  end
  return out
end

local LINKS = { Inlines = link_inlines }

local function link_cell_sections(blocks)
  -- a "Section(s)" column cell such as "5, 6, A.7" or "3, A.1–A.4"
  return pandoc.Blocks(blocks):walk { Inlines = function(inls)
    local out = pandoc.Inlines {}
    for _, el in ipairs(inls) do
      local id, rest
      if el.t == 'Str' then id, rest = match_id('cell', el.text) end
      if id then emit_id(out, 'cell', id, rest, nil) else out:insert(el) end
    end
    return out
  end }
end

-------------------------------------------------------------------------------------------------------------- tables
-- Column widths per table: a number is a fraction of \linewidth (ragged-right p column), a string a literal column.
local TABLE_SPECS = {
  ['1'] = { cols = { 0.27, 0.27, 0.30, 0.16 } },
  ['2'] = { cols = { 'l', 'l', 'l', 'l' } },
  ['3'] = { cols = { 0.235, 0.255, 0.255, 0.255 }, bracket_newline = true },
  ['4'] = { cols = { 0.10, 0.12, 0.235, 0.105, 0.105, 0.245, 0.09 }, breakable_cols = { [1] = true } },
  ['5'] = { cols = { 0.24, 0.21, 0.11, 0.21, 0.23 }, breakable_cols = { [1] = true, [5] = true } },
  ['6'] = { cols = { 0.42, 0.29, 0.29 }, breakable_cols = { [1] = true } },
  ['7'] = { cols = { 0.31, 0.23, 0.23, 0.23 }, breakable_cols = { [1] = true } },
  ['8'] = { cols = { 0.28, 0.13, 0.13, 0.24, 0.22 }, breakable_cols = { [1] = true } },
  ['9'] = { cols = { 0.20, 0.20, 0.20, 0.20, 0.20 } },
  ['corpus'] = { cols = { 0.30, 0.27, 0.19, 0.24 }, place = '!h' },
  ['A1'] = { cols = { 0.40, 0.20, 0.20, 0.20 } },
  ['A2'] = { cols = { 0.085, 0.12, 0.07, 0.255, 0.2, 0.27 } },
  ['A3'] = { cols = { 'l', 'l', 'l', 'l', 'l' } },
  ['A4'] = { cols = { 0.26, 0.14, 0.15, 0.15, 0.15, 0.15 } },
  ['A5'] = { cols = { 'l', 'l', 'l', 'l', 'l' } },
  ['A6'] = { cols = { 0.16, 0.15, 0.155, 0.18, 0.175, 0.18 }, breakable_cols = { [1] = true } },
  ['A7'] = { cols = { 0.33, 0.09, 0.58 }, long = true },
  ['A8'] = { cols = { 0.26, 0.24, 0.13, 0.24, 0.13 } },
  ['A9'] = { cols = { 0.22, 0.20, 0.14, 0.16, 0.13, 0.15 }, breakable_cols = { [1] = true } },
  ['A10'] = { cols = { 0.36, 0.14, 0.22, 0.28 }, long = true, breakable_cols = { [1] = true } },
  ['A11'] = { cols = { 0.13, 0.05, 0.06, 0.08, 0.227, 0.227, 0.226 }, breakable_cols = { [1] = true } },
  ['A12'] = { cols = { 0.18, 0.205, 0.205, 0.205, 0.205 }, breakable_cols = { [1] = true, [2] = true, [3] = true, [4] = true, [5] = true } },
  ['prereg'] = { cols = { 0.24, 0.18, 0.27, 0.23, 0.08 }, long = true },
}

local function txt(el)
  if not el then return '' end
  if el.t == 'Str' then return el.text end
  if el.t == 'Emph' or el.t == 'Strong' then return stringify(el) end
  return ''
end

local function nbsp_cell(blocks, bracket_newline)
  return pandoc.Blocks(blocks):walk { Inlines = function(inls)
    local out = pandoc.Inlines {}
    for k, el in ipairs(inls) do
      if el.t == 'Space' then
        local prev, nxt = txt(inls[k - 1]), txt(inls[k + 1])
        local num = function(s) return s:match('%d') ~= nil end
        if bracket_newline and nxt:match('^%[') then
          out:insert(raw('\\newline '))
        elseif nxt == '×' or prev == '×' or ((nxt == '→' or nxt == 'vs') and num(prev))
            or ((prev == '→' or prev == 'vs') and num(nxt)) or (prev:match('^%[') and prev:match(',$'))
            or prev == 'p' or (nxt == 'to' and num(prev)) or (prev == 'to' and num(nxt)) or prev == '/' or nxt == '/'
            or (nxt:match('^%(') and num(prev) and num(nxt)) then
          out:insert(pandoc.Str('\u{00A0}'))
        else
          out:insert(el)
        end
      else
        out:insert(el)
      end
    end
    return out
  end }
end

local function to_latex(blocks)
  local s = pandoc.write(pandoc.Pandoc(blocks), 'latex', { wrap_text = 'none' })
  return (s:gsub('%s+$', ''):gsub('\n', ' '))
end

local function colspec(cols)
  local parts = {}
  for _, c in ipairs(cols) do
    if type(c) == 'number' then
      parts[#parts + 1] = string.format('P{\\dimexpr %.3f\\linewidth-2\\tabcolsep\\relax}', c)
    else
      parts[#parts + 1] = c
    end
  end
  return '@{}' .. table.concat(parts) .. '@{}'
end

local UNNUMBERED = { ['Corpora of the gaps study.'] = 'corpus', ['Pre-registration documents.'] = 'prereg' }

local function render_table(tbl, caption, id, capname)
  local key = id or UNNUMBERED[capname]
  local spec = TABLE_SPECS[key or '?']
  if not spec then error('no column spec for table ' .. tostring(id)) end
  local ncol = #tbl.colspecs
  if #spec.cols ~= ncol then error('column count mismatch for table ' .. tostring(id)) end
  local sec_col = nil
  local function row_tex(row, is_head)
    local cells = {}
    for c, cell in ipairs(row.cells) do
      local blocks = cell.contents
      if not is_head and c == sec_col then blocks = link_cell_sections(blocks) end
      blocks = blocks:walk(LINKS)
      if not is_head and not (spec.breakable_cols and spec.breakable_cols[c]) then
        blocks = nbsp_cell(blocks, spec.bracket_newline)
      end
      cells[#cells + 1] = to_latex(blocks)
    end
    return table.concat(cells, ' & ') .. ' \\\\'
  end
  local head = tbl.head.rows[1]
  for c, cell in ipairs(head.cells) do
    local h = stringify(cell.contents)
    if h == 'Section' or h == 'Sections' then sec_col = c end
  end
  local head_tex = row_tex(head, true)
  local body = {}
  for _, b in ipairs(tbl.bodies) do
    for _, r in ipairs(b.body) do body[#body + 1] = row_tex(r, false) end
  end
  local cap = to_latex({ pandoc.Plain(caption) })
  local lab = id and ('\\label{tab:' .. id .. '}') or ''
  if id then defined['tab:' .. id] = true end
  local cs = colspec(spec.cols)
  if spec.long then
    local name = id and ('Table ' .. id) or capname:gsub('%.$', '')
    return rawb(table.concat({
      '\\needspace{6\\baselineskip}',
      '\\begingroup\\footnotesize\\setlength{\\tabcolsep}{4pt}\\setlength{\\LTcapwidth}{\\linewidth}',
      '\\begin{longtable}{' .. cs .. '}',
      '\\caption[]{' .. cap .. '}' .. lab .. '\\\\',
      '\\toprule', head_tex, '\\midrule', '\\endfirsthead',
      '\\multicolumn{' .. ncol .. '}{@{}l}{\\emph{' .. name .. ', continued}}\\\\',
      '\\toprule', head_tex, '\\midrule', '\\endhead',
      '\\bottomrule', '\\endlastfoot',
      table.concat(body, '\n'),
      '\\end{longtable}', '\\endgroup' }, '\n'))
  end
  return rawb(table.concat({
    '\\begin{table}[' .. (spec.place or 'tbp') .. ']', '\\centering\\footnotesize\\setlength{\\tabcolsep}{4pt}',
    '\\caption[]{' .. cap .. '}' .. lab,
    '\\begin{tabular}{' .. cs .. '}', '\\toprule', head_tex, '\\midrule',
    table.concat(body, '\n'), '\\bottomrule', '\\end{tabular}', '\\end{table}' }, '\n'))
end

-- split a caption paragraph's leading **Table N.** / **Figure N.** label from the rest
local function split_caption(inlines, word)
  local first = inlines[1]
  if not (first and first.t == 'Strong') then return nil end
  local s = stringify(first)
  local id = s:match('^' .. word .. ' (%w+)%.$')
  local rest = pandoc.Inlines { raw('\\textbf{' .. s .. '}') }
  for k = 2, #inlines do rest:insert(inlines[k]) end
  return id, rest, first
end

------------------------------------------------------------------------------------------------------------- figures
local FIG_SPECS = {
  ['1'] = { width = '\\linewidth', place = '!htb' },
  ['2'] = { width = '0.72\\linewidth', place = '!htb' },
  ['3'] = { width = '\\linewidth', place = '!htb' },
  ['4'] = { width = '0.85\\linewidth', place = '!htb' },
  ['5'] = { width = '0.92\\linewidth', place = '!htb' },
}

local function render_figure(fig)
  local img
  fig:walk { Image = function(im) img = im end }
  local capinl = pandoc.utils.blocks_to_inlines(fig.caption.long)
  local id, rest = split_caption(capinl, 'Figure')
  if not id then error('figure without "Figure N." caption: ' .. stringify(capinl)) end
  rest = rest:walk(LINKS)
  local spec = FIG_SPECS[id] or { width = '\\linewidth', place = '!htb' }
  defined['fig:' .. id] = true
  return rawb(table.concat({
    '\\begin{figure}[' .. spec.place .. ']', '\\centering',
    '\\includegraphics[width=' .. spec.width .. ']{' .. img.src .. '}',
    '\\caption[]{' .. to_latex({ pandoc.Plain(rest) }) .. '}\\label{fig:' .. id .. '}',
    '\\end{figure}' }, '\n'))
end

------------------------------------------------------------------------------------------------------------ headings
local counters = { sec = 0, sub = 0, appendix = false }

local function render_heading(h)
  local text = stringify(h.content)
  local inl = h.content
  local function title_from(prefix_len)
    -- drop the typed number (first Str and the following Space)
    local out = pandoc.Inlines {}
    for k = prefix_len + 1, #inl do out:insert(inl[k]) end
    return to_latex({ pandoc.Plain(out) })
  end
  if h.level == 2 then
    local num = text:match('^(%d+) ')
    local app = text:match('^Appendix (%u) ')
    if num then
      counters.sec = counters.sec + 1; counters.sub = 0
      if tonumber(num) ~= counters.sec then error('section number ' .. num .. ' out of sequence') end
      defined['sec:' .. num] = true
      return rawb('\\section{' .. title_from(2) .. '}\\label{sec:' .. num .. '}')
    elseif app then
      local pre = ''
      if not counters.appendix then
        counters.appendix = true; counters.sec = 0
        pre = "\\clearpage\n\\appendix\n\\setappendixprefix\n"
      end
      counters.sec = counters.sec + 1; counters.sub = 0
      if string.byte(app) - 64 ~= counters.sec then error('appendix ' .. app .. ' out of sequence') end
      defined['app:' .. app] = true
      return rawb(pre .. '\\section{' .. title_from(4) .. '}\\label{app:' .. app .. '}')
    else
      local t = to_latex({ pandoc.Plain(inl) })
      return rawb('\\section*{' .. t .. '}\\phantomsection\\addcontentsline{toc}{section}{' .. t .. '}')
    end
  elseif h.level == 3 then
    local num = text:match('^(%w+%.%d+) ')
    if not num then error('unnumbered subsection: ' .. text) end
    counters.sub = counters.sub + 1
    local major, minor = num:match('^(%w+)%.(%d+)$')
    local expect = counters.appendix and string.char(64 + counters.sec) or tostring(counters.sec)
    if major ~= expect or tonumber(minor) ~= counters.sub then error('subsection ' .. num .. ' out of sequence') end
    local lab = label_for('sec', num)
    defined[lab] = true
    return rawb('\\subsection{' .. title_from(2) .. '}\\label{' .. lab .. '}')
  end
  error('unexpected heading level ' .. h.level .. ': ' .. text)
end

---------------------------------------------------------------------------------------------------------- references
local function ref_key(para)
  local inl = para.content
  local first = nil
  for _, el in ipairs(inl) do
    if el.t == 'Str' then
      if el.text ~= 'de' then first = norm_name(el.text); break end
    end
  end
  for _, el in ipairs(inl) do
    if el.t == 'Str' then
      local y = el.text:match('%((%d%d%d%d%l?)%)%.?$')
      if y then return first, first .. y end
    end
  end
end

local first_reference = true
local function render_reference(para, key)
  -- the first paragraph after a heading loses its (negative) indent, so give it explicitly
  local out = pandoc.Inlines { raw((first_reference and '\\hspace*{-1.5em}' or '') .. '\\hypertarget{' .. anchor(key) .. '}{}') }
  first_reference = false
  for _, el in ipairs(para.content) do
    local id, rest
    if el.t == 'Str' then id, rest = el.text:match('^arXiv:(%d%d%d%d%.%d%d%d%d%d?)(.*)$') end
    if id then
      out:insert(raw('\\href{https://arxiv.org/abs/' .. id .. '}{arXiv:' .. id .. '}'))
      if rest ~= '' then out:insert(pandoc.Str(rest)) end
    else
      out:insert(el)
    end
  end
  return pandoc.Para(out)
end

--------------------------------------------------------------------------------------------------------------- main
function Pandoc(doc)
  doc = doc:walk { Code = Code }
  local blocks = doc.blocks
  -- locate the reference list and collect its keys first, so citations anywhere can link to it
  local ref_start, ref_end
  for k, b in ipairs(blocks) do
    if b.t == 'Header' and b.level == 2 then
      local t = stringify(b.content)
      if t == 'References' then ref_start = k
      elseif ref_start and not ref_end then ref_end = k end
    end
  end
  for k = ref_start + 1, ref_end - 1 do
    local b = blocks[k]
    if b.t == 'Para' then
      local surname, key = ref_key(b)
      if not key then error('reference without (year): ' .. stringify(b)) end
      if refkeys[key] then error('duplicate reference key ' .. key) end
      refkeys[key] = true
      refsurnames[surname] = true
    end
  end

  local out = pandoc.Blocks {}
  local k = 1
  while k <= #blocks do
    local b = blocks[k]
    if b.t == 'Header' then
      out:insert(render_heading(b))
      if k == ref_start then
        out:insert(rawb('\\begingroup\\raggedright\\small\\setlength{\\leftskip}{1.5em}'
          .. '\\setlength{\\parindent}{-1.5em}\\setlength{\\parskip}{3pt plus 1pt}'))
      end
    elseif k > ref_start and k < ref_end then
      if b.t == 'Para' then
        local _, key = ref_key(b)
        out:insert(render_reference(b, key))
      else
        out:insert(b)
      end
      if k == ref_end - 1 then out:insert(rawb('\\par\\endgroup')) end
    elseif b.t == 'Para' and blocks[k + 1] and blocks[k + 1].t == 'Table' and b.content[1]
        and b.content[1].t == 'Strong' then
      local id, rest, first = split_caption(b.content, 'Table')
      rest = rest:walk(LINKS)
      out:insert(render_table(blocks[k + 1], rest, id, stringify(first)))
      k = k + 1
    elseif b.t == 'Table' then
      error('table without a caption paragraph: ' .. stringify(b):sub(1, 80))
    elseif b.t == 'Figure' then
      out:insert(render_figure(b))
    else
      out:insert(b:walk(LINKS))
    end
    k = k + 1
  end

  local missing = {}
  for lab in pairs(used) do if not defined[lab] then missing[#missing + 1] = lab end end
  if #missing > 0 then
    table.sort(missing)
    error('cross-references without a target: ' .. table.concat(missing, ', '))
  end
  doc.blocks = out
  return doc
end
