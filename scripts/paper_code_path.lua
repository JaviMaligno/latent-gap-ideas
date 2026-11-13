-- Inline code -> \path (breakable with xurl) for the arXiv LaTeX build (scripts/build_paper_latex.py).
local delims = {"|", "!", "+", "@", "=", "^"}
function Code(el)
  for _, d in ipairs(delims) do
    if not el.text:find(d, 1, true) then
      return pandoc.RawInline("latex", "\\path" .. d .. el.text .. d)
    end
  end
  return el
end
